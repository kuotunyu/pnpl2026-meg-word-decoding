"""Sentence-context word decoder (in the spirit of d'Ascoli et al. 2025).

Per-word CNN encoder on [onset, onset+1 s] PCA epochs -> Transformer over the words of a sentence
(word order only; no onset-timing features) -> per-word logits over the BIG most frequent training
words, of which classes 0..49 are the competition vocabulary.

All word epochs of all runs live on the GPU as float16 (~16 GB), filled file by file.
  python train_ctx.py --tag c1 [--full] [--epochs 20] [--big 1000] [--maxlen 12]
"""
import argparse, glob, json, os, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import TARGET, VAL_RUNS, Net, bacc_topk, corpus_of

HERE = Path(__file__).resolve().parent
DER = HERE.parents[2] / "data" / "derived" / "subj0_pca_epochs"
OUTS = HERE.parent / "outputs"


def sentences(sent, onset):
    """Consecutive words sharing a sentence index (and no >2 s gap)."""
    ch = np.r_[True, (np.diff(sent) != 0) | (np.diff(onset) > 2.0)]
    starts = np.nonzero(ch)[0]; ends = np.r_[starts[1:], len(sent)]
    return list(zip(starts.tolist(), ends.tolist()))


def segments(sent, onset, maxlen):
    """Sentences -> chunks of <= maxlen words."""
    out = []
    for s, e in sentences(sent, onset):
        for c in range(s, e, maxlen):
            out.append((c, min(e, c + maxlen)))
    return out


def load(full, big, maxlen, dedup=True):
    """Sentences whose exact text occurs more than once in training (TIMIT SA, MOCHA, ...) are
    trained as isolated words so the Transformer cannot memorise them. Validation also reports
    `novel`: only sentences whose text never occurs in training."""
    files = sorted(glob.glob(str(DER / "runs" / "*.npz")))
    names = [os.path.basename(f)[:-4] for f in files]
    is_val = [n in VAL_RUNS and not full for n in names]
    cnt = Counter(); txt = Counter()
    metas = []
    for f, v in zip(files, is_val):
        d = np.load(f, allow_pickle=True)
        w = d["word"].tolist()
        metas.append((w, d["sentence"], d["onset"]))
        if not v:
            cnt.update(w)
            txt.update(" ".join(w[a:b]) for a, b in sentences(d["sentence"], d["onset"]))
    classes = list(TARGET) + [w for w, _ in cnt.most_common() if w and w not in set(TARGET)][: big - 50]
    cid = {w: i for i, w in enumerate(classes)}
    N = sum(len(m[0]) for m in metas)
    X = torch.empty((N,) + tuple(np.load(files[0], allow_pickle=True)["X"].shape[1:]), dtype=torch.float16, device="cuda")
    Y = torch.full((N,), -100, dtype=torch.long)
    tr_seg, va_seg, va_info = [], [], []
    off = 0; n_iso = 0
    for f, n, v, (w, sent, on) in zip(files, names, is_val, metas):
        x = np.load(f, allow_pickle=True)["X"]
        X[off:off + len(x)] = torch.from_numpy(x).cuda()
        Y[off:off + len(x)] = torch.tensor([cid.get(t, -100) for t in w])
        for a, b in sentences(sent, on):
            key = " ".join(w[a:b])
            if v:
                for c in range(a, b, maxlen):
                    va_seg.append((c + off, min(b, c + maxlen) + off)); va_info.append(txt[key] == 0)
            elif dedup and txt[key] > 1 and b - a > 1:
                tr_seg += [(t + off, t + off + 1) for t in range(a, b)]; n_iso += b - a
            else:
                tr_seg += [(c + off, min(b, c + maxlen) + off) for c in range(a, b, maxlen)]
        off += len(x)
    print("words trained without context (repeated sentences):", n_iso, flush=True)
    return X, Y.cuda(), tr_seg, va_seg, va_info, classes


class CtxNet(nn.Module):
    def __init__(self, nc, width=320, depth=6, drop=0.3, d=512, layers=4, maxlen=16, sem_dim=0):
        super().__init__()
        self.sem_dim = sem_dim
        if sem_dim:  # semantic head: cosine to fixed T5 word embeddings (d'Ascoli-style target)
            self.semp = nn.Linear(d, sem_dim)
            self.register_buffer("E", torch.zeros(nc, sem_dim))
            self.sscale = nn.Parameter(torch.tensor(2.3))
        enc = Net(80, width, nc, drop, depth)
        self.body = nn.Sequential(enc.inp, enc.body, enc.pool, nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(width * 16, d), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(maxlen, d))
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers)
        self.local = nn.Linear(d, nc)   # context-free head (keeps per-word features discriminative)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, nc))

    prep = ""  # "demean": subtract each word window's per-channel mean over its non-zero (recorded) samples

    def forward(self, x, pad):
        """x: (B, L, 80, 125); pad: (B, L) True where padded."""
        B, L = pad.shape
        if self.prep in ("demean", "inorm", "gnorm"):
            v = (x != 0).to(x.dtype)
            n = v.sum(-1, keepdim=True).clamp(min=1)
            x = (x - (x * v).sum(-1, keepdim=True) / n) * v
            if self.prep == "inorm":  # instance norm: per-window per-channel std (robust to normalisation shift)
                x = x / ((x * x).sum(-1, keepdim=True) / n).sqrt().clamp(min=1e-3) * v
            elif self.prep == "gnorm":  # one scale per window (keeps relative channel amplitudes)
                nn_ = v.sum((-2, -1), keepdim=True).clamp(min=1)
                x = x / ((x * x).sum((-2, -1), keepdim=True) / nn_).sqrt().clamp(min=1e-3) * v
        h = self.body(x.reshape(B * L, *x.shape[2:])).reshape(B, L, -1)
        loc = self.local(h)
        h = self.tf(h + self.pos[:L], src_key_padding_mask=pad)
        self.last_h = h
        return self.out(h), loc

    def sem_logits(self, h):
        return F.normalize(self.semp(h), dim=-1) @ self.E.T * self.sscale.exp()


VOCABCTX = 0.0   # --vocabctx: probability that a training span keeps only its 50-vocabulary words (holdout-like context)
ISVOC = None     # CPU bool array: word is one of the 50 vocabulary words


def batch_from(segs, X, Y, train, g, dropword=0.0):
    lens = []; idx = []
    for a, b in segs:
        n = b - a
        if train:
            k = int(torch.randint(1, n + 1, (1,), generator=g))  # random sub-span length incl. single words
            s = a + int(torch.randint(0, n - k + 1, (1,), generator=g))
            span = list(range(s, s + k))
            if dropword > 0 and k > 1:
                keep = (torch.rand(k, generator=g) > dropword).tolist()
                span = [t for t, kk in zip(span, keep) if kk] or span[:1]
            if VOCABCTX > 0 and k > 1 and float(torch.rand(1, generator=g)) < VOCABCTX:
                span = [t for t in span if ISVOC[t]] or span[:1]
            idx.append(span)
        else:
            idx.append(list(range(a, b)))
    L = max(len(i) for i in idx)
    I = torch.zeros((len(idx), L), dtype=torch.long)
    pad = torch.ones((len(idx), L), dtype=torch.bool)
    for r, i in enumerate(idx):
        I[r, :len(i)] = torch.tensor(i); pad[r, :len(i)] = False
    I, pad = I.cuda(), pad.cuda()
    return X[I], torch.where(pad, torch.full_like(Y[I], -100), Y[I]), pad


FS = int(os.environ.get("PNPL_FS", "50"))  # sample rate of long-window data: onset at FS//2, 1 s = FS samples
ON, OFF = FS // 2, FS // 2 + FS


def augment(x, g, w3=False, iso=False):
    x = x.float().clamp(-20, 20)
    B, L = x.shape[:2]
    x = x * (1 + 0.1 * torch.randn(B, L, 1, 1, device=x.device))
    x = x + 0.1 * torch.randn_like(x)
    x = x * (torch.rand(B, L, x.shape[2], 1, device=x.device) > 0.1).to(x.dtype)
    if w3:  # long windows (50 Hz, onset at sample 25): mimic what the holdout cannot provide
        T = x.shape[-1]; t = torch.arange(T, device=x.device)
        r = torch.rand(B, L, 3, device=x.device)
        keep = torch.ones(B, L, T, device=x.device)
        keep = keep * torch.where((r[..., 0:1] < 0.25), (t >= ON).float(), torch.ones_like(keep))      # no pre-onset (sentence start)
        cut = OFF + (torch.rand(B, L, 1, device=x.device) * (T - OFF)).long()
        keep = keep * torch.where((r[..., 1:2] < 0.4), (t < cut).float(), torch.ones_like(keep))      # sentence ends early
        isow = (r[..., 2:3] < 0.1)
        keep = keep * torch.where(isow, ((t >= ON) & (t < OFF)).float(), torch.ones_like(keep))         # isolated word: [0, 1 s]
        x = x * keep[:, :, None, :]
    if iso:  # isolated-word specialist: only [onset, onset + 1 s] (samples 25..75 at 50 Hz) is ever seen
        x[..., :ON] = 0; x[..., OFF:] = 0
    return x


SEM_OUT = []
EVAL_KEEP = None  # (lo, hi): zero everything outside these time samples at evaluation (isolated-word condition)


@torch.no_grad()
def run_eval(model, segs, X, Y, bs=256):
    model.eval(); P, Q, T = [], [], []
    for i in range(0, len(segs), bs):
        xb, yb, pad = batch_from(segs[i:i + bs], X, Y, False, None)
        xb = xb.float().clamp(-20, 20)
        if EVAL_KEEP is not None:
            xb[..., :EVAL_KEEP[0]] = 0; xb[..., EVAL_KEEP[1]:] = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo, loc = model(xb, pad)
        m = ~pad
        lp = F.log_softmax(lo.float()[m][:, :50], 1)
        if getattr(model, "sem_dim", 0):
            sl = F.log_softmax(model.sem_logits(model.last_h.float())[m][:, :50], 1)
            SEM_OUT.append(sl.cpu())
            lp = 0.5 * (lp + sl)
        P.append(lp.cpu()); Q.append(F.log_softmax(loc.float()[m][:, :50], 1).cpu())
        T.append(yb[m].cpu())
    return torch.cat(P).numpy(), torch.cat(Q).numpy(), torch.cat(T).numpy()


def holdout_segments():
    raw = HERE.parents[2] / "data" / "raw" / "LibriBrain-Competition-2026" / "COMPETITION_HOLDOUT"
    try:
        wm = np.load(raw / "subj00_holdout01_sentence.npz")["word_mask"]
    except Exception:  # no / partial Hugging Face download (rate limit): same mask saved in sources/holdout_wordmask.csv
        wm = np.loadtxt(HERE / "holdout_wordmask.csv", delimiter=",").astype(bool)
    segs, off = [], 0
    for r in wm:
        n = int(r.sum()); segs.append((off, off + n)); off += n
    nw = 92
    segs += [(off + i, off + i + 1) for i in range(nw)]
    return segs, off + nw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--big", type=int, default=1000)
    ap.add_argument("--maxlen", type=int, default=12)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--wpow", type=float, default=0.3)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dropword", type=float, default=0.0)
    ap.add_argument("--prep", default="", help="'demean' per-window per-channel mean removal; 'inorm' + per-channel std; 'gnorm' + one std per window")
    ap.add_argument("--ema", type=float, default=0.0, help="keep an exponential moving average of the weights (e.g. 0.999); used for eval and saved")
    ap.add_argument("--vocabctx", type=float, default=0.0, help="prob. a training span keeps only vocabulary words")
    ap.add_argument("--nodedup", action="store_true")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--cdepth", type=int, default=6)
    ap.add_argument("--sem", type=float, default=0.0)
    ap.add_argument("--data", default="subj0_pca_epochs", help="folder under data/derived")
    ap.add_argument("--w3aug", action="store_true")
    ap.add_argument("--iso", action="store_true", help="isolated-word specialist: single-word sequences, [0,1 s] only")
    a = ap.parse_args()
    global DER
    DER = DER.parent / a.data
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = OUTS / a.tag; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, Y, tr, va, va_info, classes = load(a.full, a.big, a.maxlen, not a.nodedup)
    va_novel = [s for s, nv in zip(va, va_info) if nv]
    if a.iso:
        global EVAL_KEEP
        EVAL_KEEP = (ON, OFF)
        tr = [(t, t + 1) for s0, e0 in tr for t in range(s0, e0)]
        va = [(t, t + 1) for s0, e0 in va for t in range(s0, e0)]
        va_novel = va
        print("iso mode: single-word segments", len(tr), "eval window samples 25..75", flush=True)
    NC = len(classes)
    global VOCABCTX, ISVOC
    VOCABCTX = a.vocabctx; ISVOC = ((Y >= 0) & (Y < 50)).cpu().numpy()
    print(f"words {len(X)} train segs {len(tr)} val segs {len(va)} classes {NC} load {time.time() - t0:.0f}s", flush=True)
    ytr = torch.cat([Y[s:e] for s, e in tr]); ytr = ytr[ytr >= 0]
    prior = torch.bincount(ytr, minlength=NC).float() + 1; prior /= prior.sum()
    w = prior ** (-a.wpow); w = w / (w * prior).sum()
    prior50 = (prior[:50] / prior[:50].sum()).cpu().numpy()
    model = CtxNet(NC, width=a.width, depth=a.cdepth, d=a.d, layers=a.layers, maxlen=max(a.maxlen, 16),
                   sem_dim=1024 if a.sem > 0 else 0).cuda()
    if a.sem > 0:
        t5 = np.load(DER.parent / "t5_word_emb.npz")
        tix = {w: i for i, w in enumerate(t5["words"].tolist())}
        E = np.zeros((NC, 1024), np.float32)
        for i, word in enumerate(classes):
            if word in tix:
                E[i] = t5["emb"][tix[word]]
        model.E.copy_(torch.from_numpy(E))
        print("T5 embeddings found for", int((np.abs(E).sum(1) > 0).sum()), "of", NC, "classes", flush=True)
    model.prep = a.prep
    print("params", round(sum(p.numel() for p in model.parameters()) / 1e6, 2), "M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    nb = len(tr) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * nb, pct_start=0.1)
    g = torch.Generator().manual_seed(a.seed)
    ema = None
    if a.ema > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(a.ema), use_buffers=True)
    hist = []
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(tr), generator=g).tolist(); tl = 0
        for bi in range(nb):
            segs = [tr[j] for j in perm[bi * a.bs:(bi + 1) * a.bs]]
            xb, yb, pad = batch_from(segs, X, Y, True, g, a.dropword)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo, loc = model(augment(xb, g, a.w3aug, a.iso), pad)
            yf = yb.reshape(-1)
            loss = F.cross_entropy(lo.float().reshape(-1, NC), yf, weight=w, label_smoothing=a.ls, ignore_index=-100) \
                + 0.5 * F.cross_entropy(loc.float().reshape(-1, NC), yf, weight=w, label_smoothing=a.ls, ignore_index=-100)
            if a.sem > 0:
                sl = model.sem_logits(model.last_h.float()).reshape(-1, NC)
                loss = loss + a.sem * F.cross_entropy(sl, yf, ignore_index=-100)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tl += loss.item()
            if ema is not None:
                ema.update_parameters(model)
        rec = {"epoch": ep + 1, "loss": round(tl / nb, 4), "sec": round(time.time() - t0)}
        evm = ema.module if ema is not None else model
        if va and (ep % 2 == 1 or ep == a.epochs - 1):
            P, Q, T = run_eval(evm, va, X, Y)
            m = (T >= 0) & (T < 50)
            rec["val_ctx"] = round(bacc_topk(P[m], T[m]), 4); rec["val_local"] = round(bacc_topk(Q[m], T[m]), 4)
            P, Q, T = run_eval(evm, va_novel, X, Y); m = (T >= 0) & (T < 50)
            rec["novel_ctx"] = round(bacc_topk(P[m], T[m]), 4); rec["novel_local"] = round(bacc_topk(Q[m], T[m]), 4)
        hist.append(rec); print(json.dumps(rec), flush=True)
    if ema is not None:
        model = ema.module
    torch.save({"state": model.state_dict(), "args": vars(a), "classes": classes, "prior": prior.cpu()}, out / "model.pt")
    res = {"args": vars(a), "hist": hist}
    if va:
        P, Q, T = run_eval(model, va, X, Y)
        m = (T >= 0) & (T < 50)
        np.savez(out / "val_logits.npz", ctx=P[m], local=Q[m], y=T[m], prior=prior50)
        res["val_ctx"] = bacc_topk(P[m], T[m]); res["val_local"] = bacc_topk(Q[m], T[m])
        P, Q, T = run_eval(model, va_novel, X, Y); m = (T >= 0) & (T < 50)
        np.savez(out / "val_novel_logits.npz", ctx=P[m], local=Q[m], y=T[m], prior=prior50)
        res["novel_ctx"] = bacc_topk(P[m], T[m]); res["novel_local"] = bacc_topk(Q[m], T[m])
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    # holdout
    del X
    h = np.load(DER / "holdout_subj00.npz")
    Xh = torch.from_numpy(h["X"]).cuda()
    hs, n = holdout_segments(); assert n == len(Xh)
    Yh = torch.zeros(n, dtype=torch.long, device="cuda")
    P, Q, _ = run_eval(model, hs, Xh, Yh)
    np.savez(out / "holdout_logits.npz", logp=P, local=Q, source=h["source"], prior=prior50)
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    print("DONE", json.dumps({k: res[k] for k in res if k.startswith(("val", "novel"))}), round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
