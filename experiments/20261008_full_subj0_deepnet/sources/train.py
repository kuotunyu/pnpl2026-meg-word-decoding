"""Train a word classifier on compact subject-0 PCA epochs (see preprocess.py).

All selected epochs live on the GPU as float16; one process, no data loader.
  python train.py --tag v1 [--full] [--epochs 30] [--seed 0] ...
Writes outputs/<tag>/{model.pt, val_logits.npz, holdout_logits.npz, metrics.json}.
"""
import argparse, glob, json, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
COMP = HERE.parents[2]
DER = COMP / "data" / "derived" / "subj0_pca_epochs"
OUTS = HERE.parent / "outputs"

# validation runs: official LibriBrain val/test Sherlock1 sessions + held-out sentence-corpus sessions
VAL_RUNS = {
    "sub-0_ses-11_task-Sherlock1_run-2", "sub-0_ses-12_task-Sherlock1_run-2",
    "sub-0_ses-13_task-TIMIT_run-1", "sub-0_ses-14_task-TIMIT_run-1",
    "sub-0_ses-4_task-MOCHATIMIT_run-1",
    "sub-0_ses-29_task-TheMoth_run-1", "sub-0_ses-30_task-TheMoth_run-1",
}


def corpus_of(name):
    return name.split("task-")[1].split("_")[0].rstrip("0123456789") if "Sherlock" in name else name.split("task-")[1].split("_")[0]


TARGET = ["is", "the", "a", "to", "it", "i", "not", "was", "we", "be", "he", "that", "have", "this", "they", "of",
          "there", "and", "are", "in", "but", "will", "so", "all", "my", "for", "she", "were", "any", "really", "at",
          "out", "our", "am", "it's", "had", "him", "an", "very", "has", "do", "can", "time", "think", "good",
          "always", "new", "people", "as", "on"]


def load(full, big=0):
    """Returns train/val lists of (X, y, name). y indexes the class list; classes 0..49 are the targets.
    With big>0, every word among the `big` most frequent training words is kept as an extra class."""
    files = sorted(glob.glob(str(DER / "runs" / "*.npz")))
    is_val = lambda n: n in VAL_RUNS and not full
    classes = list(TARGET)
    if big:
        from collections import Counter
        cnt = Counter()
        for f in files:
            if not is_val(os.path.basename(f)[:-4]):
                cnt.update(np.load(f, allow_pickle=True)["word"].tolist())
        extra = [w for w, _ in cnt.most_common() if w not in set(TARGET) and w][: max(0, big - 50)]
        classes += extra
    cid = {w: i for i, w in enumerate(classes)}
    tr, va = [], []
    for f in files:
        name = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        y = np.array([cid.get(w, -1) for w in d["word"].tolist()], np.int64)
        if is_val(name):
            y = np.where(y < 50, y, -1)
        keep = y >= 0
        (va if is_val(name) else tr).append((d["X"][keep], y[keep], name))
    return tr, va, classes


def bacc_topk(logits, y, k=10, n=50):
    top = np.argsort(-logits, 1)[:, :k]
    hit = (top == y[:, None]).any(1)
    rec = [hit[y == c].mean() for c in range(n) if (y == c).any()]
    return float(np.mean(rec))


class Block(nn.Module):
    def __init__(self, c, dil, p):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 3, padding=dil, dilation=dil)
        self.b1 = nn.BatchNorm1d(c)
        self.c2 = nn.Conv1d(c, c, 3, padding=1)
        self.b2 = nn.BatchNorm1d(c)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = F.gelu(self.b1(self.c1(x)))
        h = self.b2(self.c2(self.drop(h)))
        return F.gelu(x + h)


class Net(nn.Module):
    def __init__(self, k_in=80, c=320, n_cls=50, p=0.3, depth=6):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(k_in, c, 1), nn.BatchNorm1d(c), nn.GELU())
        blocks = []
        for i in range(depth):
            blocks.append(Block(c, 2 ** (i % 4), p))
            if i in (1, 3):
                blocks.append(nn.Sequential(nn.Conv1d(c, c, 3, stride=2, padding=1), nn.BatchNorm1d(c), nn.GELU()))
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(c * 16, 512), nn.GELU(),
                                  nn.Dropout(0.5), nn.Linear(512, n_cls))

    def forward(self, x):
        return self.head(self.pool(self.body(self.inp(x))))


def augment(x, g):
    B, C, T = x.shape
    # random time shift by up to +-3 samples (24 ms), edge padded
    s = int(torch.randint(-3, 4, (1,), generator=g).item())
    if s:
        x = torch.roll(x, s, 2)
        if s > 0:
            x[:, :, :s] = x[:, :, s:s + 1]
        else:
            x[:, :, s:] = x[:, :, s - 1:s]
    x = x * (1 + 0.1 * torch.randn(B, 1, 1, device=x.device))
    x = x + 0.1 * torch.randn_like(x)
    mask = (torch.rand(B, C, 1, device=x.device) > 0.1).to(x.dtype)
    return x * mask


def predict(model, X, bs=4096, tta=True):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].float().clamp(-20, 20)
            lo = F.log_softmax(model(xb).float()[:, :50], 1)
            if tta:
                for s in (-2, 2):
                    lo = lo + F.log_softmax(model(torch.roll(xb, s, 2)).float()[:, :50], 1)
                lo = lo / 3
            out.append(lo.cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--wpow", type=float, default=0.5, help="class weight = prior^-wpow")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--big", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = OUTS / a.tag; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr, va, classes = load(a.full, a.big)
    NC = len(classes)
    Xtr = torch.from_numpy(np.concatenate([t[0] for t in tr])).cuda()
    ytr = torch.from_numpy(np.concatenate([t[1] for t in tr])).cuda()
    print(f"train {tuple(Xtr.shape)} runs={len(tr)} load {time.time() - t0:.0f}s", flush=True)
    if va:
        Xva = torch.from_numpy(np.concatenate([t[0] for t in va])).cuda()
        yva = np.concatenate([t[1] for t in va])
        cva = np.concatenate([[corpus_of(t[2])] * len(t[1]) for t in va])
        print("val", tuple(Xva.shape), {c: int((cva == c).sum()) for c in np.unique(cva)}, flush=True)
    prior = torch.bincount(ytr, minlength=NC).float() + 1
    prior = prior / prior.sum()
    w = prior ** (-a.wpow); w = w / (w * prior).sum()
    model = Net(Xtr.shape[1], a.width, NC, a.drop, a.depth).cuda()
    prior50 = (prior[:50] / prior[:50].sum()).cpu().numpy()
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(Xtr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(a.seed)
    hist = []
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(Xtr), device="cuda"); tl = 0; nb = 0
        for i in range(0, len(Xtr) - a.bs + 1, a.bs):
            idx = perm[i:i + a.bs]
            xb = augment(Xtr[idx].float().clamp(-20, 20), g); yb = ytr[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
            loss = F.cross_entropy(logits.float(), yb, weight=w, label_smoothing=a.ls)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tl += loss.item(); nb += 1
        rec = {"epoch": ep + 1, "loss": tl / nb, "sec": round(time.time() - t0)}
        if va and (ep % 3 == 2 or ep == a.epochs - 1):
            lv = predict(model, Xva, tta=False) - 0.0
            adj = lv - np.log(prior50)[None] * 1.0
            rec["val_bacc10"] = bacc_topk(lv, yva); rec["val_bacc10_adj"] = bacc_topk(adj, yva)
            rec["by_corpus_adj"] = {c: round(bacc_topk(adj[cva == c], yva[cva == c]), 4) for c in np.unique(cva)}
        hist.append(rec); print(json.dumps(rec), flush=True)
    torch.save({"state": model.state_dict(), "args": vars(a), "prior": prior.cpu(), "classes": classes}, out / "model.pt")
    res = {"args": vars(a), "hist": hist, "n_train": int(len(Xtr))}
    if va:
        lv = predict(model, Xva)
        np.savez(out / "val_logits.npz", logp=lv, y=yva, corpus=cva, prior=prior50)
        for tau in (0, 0.5, 1.0, 1.5):
            res[f"val_tta_bacc10_tau{tau}"] = bacc_topk(lv - tau * np.log(prior50)[None], yva)
    hf = DER / "holdout_subj00.npz"
    if hf.exists():
        h = np.load(hf)
        lh = predict(model, torch.from_numpy(h["X"]).cuda())
        np.savez(out / "holdout_logits.npz", logp=lh, source=h["source"], prior=prior50)
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    print("DONE", json.dumps({k: v for k, v in res.items() if k.startswith("val_tta")}), round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
