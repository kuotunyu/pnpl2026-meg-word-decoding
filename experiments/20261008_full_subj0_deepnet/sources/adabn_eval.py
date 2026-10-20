"""AdaBN check: re-estimate the CNN encoder's BatchNorm statistics on unlabeled target-session data, then
evaluate. Validation version: target = the held-out validation runs (other sessions than most training data);
reports novel-sentence BAcc@10 and holdout-like (proxy) BAcc@10 before and after.
  python adabn_eval.py <tag> [cpu|cuda] [data_dir_name]
"""
import copy, glob, os, sys
from collections import Counter
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import train_ctx as T
from train import TARGET, VAL_RUNS, bacc_topk

tag = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
data = sys.argv[3] if len(sys.argv) > 3 else "subj0_pca_epochs"
DER = T.DER.parent / data
files = sorted(glob.glob(str(DER / "runs" / "*.npz")))
txt = Counter()
for f in files:
    if os.path.basename(f)[:-4] not in VAL_RUNS:
        d = np.load(f, allow_pickle=True); w = d["word"].tolist()
        txt.update(" ".join(w[a:b]) for a, b in T.sentences(d["sentence"], d["onset"]))
cid = {w: i for i, w in enumerate(TARGET)}
ck = torch.load(T.OUTS / tag / "model.pt", map_location=dev, weights_only=False)
a = ck["args"]
base = T.CtxNet(len(ck["classes"]), width=a.get("width", 320), depth=a.get("cdepth", 6), d=a.get("d", 512),
                layers=a["layers"], maxlen=max(a["maxlen"], 16)).to(dev)
base.load_state_dict(ck["state"]); base.prep = a.get("prep", ""); base.eval()
prior = (ck["prior"][:50] / ck["prior"][:50].sum()).cpu().numpy()

# validation material: all val words (for BN stats), novel sentences (<=12-word chunks), proxy segments
allX, novel, proxy = [], [], []
for f in files:
    if os.path.basename(f)[:-4] not in VAL_RUNS:
        continue
    d = np.load(f, allow_pickle=True); w = d["word"].tolist(); X = d["X"]
    allX.append(X)
    for s, e in T.sentences(d["sentence"], d["onset"]):
        if txt[" ".join(w[s:e])] > 0:
            continue
        for c in range(s, e, 12):
            novel.append((X[c:min(e, c + 12)], [cid.get(x, -1) for x in w[c:min(e, c + 12)]]))
        t = s
        while t < e:
            if w[t] not in cid:
                t += 1; continue
            u = t
            while u < e and w[u] in cid:
                u += 1
            if u - t >= 3:
                for c in range(t, u, 12):
                    proxy.append((X[c:min(u, c + 12)], [cid[x] for x in w[c:min(u, c + 12)]]))
            t = u
allX = np.concatenate(allX)


@torch.no_grad()
def score(m, segs):
    P, Y = [], []
    for x, y in segs:
        xb = torch.from_numpy(x).float().clamp(-20, 20)[None].to(dev)
        lo, _ = m(xb, torch.zeros(1, len(y), dtype=torch.bool, device=dev))
        P.append(F.log_softmax(lo[0, :, :50].float(), 1).cpu().numpy()); Y += y
    P, Y = np.concatenate(P), np.array(Y); k = Y >= 0
    return [round(bacc_topk(P[k] - t * np.log(prior), Y[k]), 4) for t in (0.0, 0.3, 0.5)]


@torch.no_grad()
def adapt(m, X, bs=512):
    m = copy.deepcopy(m)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.reset_running_stats(); mod.momentum = None; mod.train()
    idx = np.random.default_rng(0).permutation(len(X))
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[idx[i:i + bs]]).float().clamp(-20, 20).to(dev)
        m.body(xb)
    m.eval(); return m


print(tag, "base   novel tau0/0.3/0.5", score(base, novel), " proxy", score(base, proxy), flush=True)
ad = adapt(base, allX)
print(tag, "adabn  novel tau0/0.3/0.5", score(ad, novel), " proxy", score(ad, proxy), flush=True)
