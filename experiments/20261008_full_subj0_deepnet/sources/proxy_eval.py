"""Holdout-like validation ("proxy"): the holdout sentence rows are sentences made only of the 50 vocabulary
words, and the model sees only those words as context. Mimic that on the validation runs: inside every novel
validation sentence take each maximal run of >= MINLEN consecutive vocabulary words, and feed that run alone
as the model's context. Saves per-word log-probs so post-processing (tau, LM fusion, ensembling) can be tuned
offline instead of on the noisy Public split.
  python proxy_eval.py <tag> [cpu|cuda] [data_dir_name]  -> outputs/<tag>/proxy_logits.npz
"""
import glob, os, sys
from collections import Counter
import numpy as np, torch, torch.nn.functional as F
import train_ctx as T
from train import TARGET, VAL_RUNS, bacc_topk

MINLEN = 3
SHIFTS = [int(v) for v in os.environ.get("PNPL_TTA", "0").split(",")]  # test-time time shifts (samples), averaged
SUFFIX = os.environ.get("PNPL_TTA_TAG", "")
tag = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
data = sys.argv[3] if len(sys.argv) > 3 else "subj0_pca_epochs"
DER = T.DER.parent / data
files = sorted(glob.glob(str(DER / "runs" / "*.npz")))
txt = Counter()
for f in files:
    if os.path.basename(f)[:-4] in VAL_RUNS:
        continue
    d = np.load(f, allow_pickle=True); w = d["word"].tolist()
    txt.update(" ".join(w[a:b]) for a, b in T.sentences(d["sentence"], d["onset"]))
cid = {w: i for i, w in enumerate(TARGET)}

ck = torch.load(T.OUTS / tag / "model.pt", map_location=dev, weights_only=False)
a = ck["args"]
m = T.CtxNet(len(ck["classes"]), width=a.get("width", 320), depth=a.get("cdepth", 6), d=a.get("d", 512),
             layers=a["layers"], maxlen=max(a["maxlen"], 16)).to(dev)
m.load_state_dict(ck["state"]); m.prep = a.get("prep", ""); m.eval()
P, Q, Yall, words, seglen = [], [], [], [], []
with torch.no_grad():
    for f in files:
        if os.path.basename(f)[:-4] not in VAL_RUNS:
            continue
        d = np.load(f, allow_pickle=True); w = d["word"].tolist(); X = d["X"]
        for s, e in T.sentences(d["sentence"], d["onset"]):
            if txt[" ".join(w[s:e])] > 0:
                continue  # not a novel sentence
            t = s
            while t < e:
                if w[t] not in cid:
                    t += 1; continue
                u = t
                while u < e and w[u] in cid:
                    u += 1
                if u - t >= MINLEN:
                    for c in range(t, u, 12):  # model maxlen
                        c2 = min(u, c + 12)
                        x0 = torch.from_numpy(X[c:c2]).float().clamp(-20, 20)[None].to(dev)
                        ps, qs = [], []
                        for sh in SHIFTS:  # shift right by sh samples (edge-padded)
                            xb = x0 if sh == 0 else torch.roll(x0, sh, -1)
                            if sh > 0: xb[..., :sh] = x0[..., :1]
                            if sh < 0: xb[..., sh:] = x0[..., -1:]
                            lo, loc = m(xb, torch.zeros(1, c2 - c, dtype=torch.bool, device=dev))
                            ps.append(F.log_softmax(lo[0, :, :50].float(), 1)); qs.append(F.log_softmax(loc[0, :, :50].float(), 1))
                        P.append(torch.stack(ps).mean(0).cpu().numpy()); Q.append(torch.stack(qs).mean(0).cpu().numpy())
                        Yall += [cid[x] for x in w[c:c2]]; words += w[c:c2]; seglen.append(c2 - c)
                t = u
P, Q, Y = np.concatenate(P), np.concatenate(Q), np.array(Yall)
prior = (ck["prior"][:50] / ck["prior"][:50].sum()).cpu().numpy()
np.savez(T.OUTS / tag / f"proxy_logits{SUFFIX}.npz", ctx=P, local=Q, y=Y, seglen=np.array(seglen), prior=prior,
         words=np.array(words))
print(tag, "proxy words", len(Y), "segments", len(seglen), "classes present", len(set(Y.tolist())))
for tau in (0.0, 0.3, 0.5, 0.7):
    print(f"{tag} proxy tau{tau}: ctx {bacc_topk(P - tau * np.log(prior), Y):.4f}", flush=True)
