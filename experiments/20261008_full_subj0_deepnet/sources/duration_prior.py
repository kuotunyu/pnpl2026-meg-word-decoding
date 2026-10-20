"""Word-timing evidence: the gap from a word's onset to the next word's onset (holdout gives word_onsets_s)
is never used by the MEG models (they see only word order). Short function words ("a", "the") have short gaps.
Fit log P(gap bin | word) - log P(gap bin) on the training runs' 50 vocabulary words (next word in the same
sentence; last word of a sentence -> no evidence), add beta * that to the log-probs.
  python duration_prior.py proxy --tags r4_ema r5_ep60 ...      (validation: proxy words, val runs held out)
  python duration_prior.py stats                                  (gap distributions train vs holdout)
"""
import glob, os, sys
import numpy as np
import train_ctx as T
from train import TARGET, VAL_RUNS, bacc_topk
from collections import Counter

DER = T.DER.parent / "subj0_pca_epochs"
files = sorted(glob.glob(str(DER / "runs" / "*.npz")))
cid = {w: i for i, w in enumerate(TARGET)}
NB = 16


NORM = os.environ.get("DUR_NORM", "none")   # none | sent: divide each gap by its sentence's median gap


def norm_gaps(g):
    if NORM == "sent":
        ok = np.isfinite(g) & (g > 0)
        if ok.sum() >= 2:
            return g / np.median(g[ok])
        return np.full_like(g, np.nan)
    return g


def gaps_of(d):
    on = d["onset"]; g = np.full(len(on), np.nan)
    for s, e in T.sentences(d["sentence"], on):
        g[s:e - 1] = norm_gaps(on[s + 1:e] - on[s:e - 1])
    return g


def table(exclude_val):
    G, C = [], []
    for f in files:
        if exclude_val and os.path.basename(f)[:-4] in VAL_RUNS:
            continue
        d = np.load(f, allow_pickle=True); g = gaps_of(d)
        for w, x in zip(d["word"].tolist(), g):
            if w in cid and np.isfinite(x) and x > 0:
                G.append(x); C.append(cid[w])
    G, C = np.log(np.array(G)), np.array(C)
    edges = np.quantile(G, np.linspace(0, 1, NB + 1)[1:-1])
    b = np.digitize(G, edges)
    cnt = np.zeros((50, NB)); np.add.at(cnt, (C, b), 1)
    pb = cnt.sum(0) / cnt.sum()
    pbc = (cnt + 5 * pb) / (cnt.sum(1, keepdims=True) + 5)       # smoothed P(bin | word)
    return edges, np.log(pbc) - np.log(pb)                      # (50, NB) log-likelihood ratio


def ev(edges, llr, gaps):
    out = np.zeros((len(gaps), 50))
    ok = np.isfinite(gaps) & (gaps > 0)
    out[ok] = llr[:, np.digitize(np.log(gaps[ok]), edges)].T
    return out


if sys.argv[1] == "stats":
    edges, llr = table(False)
    tr = []
    for f in files:
        d = np.load(f, allow_pickle=True); g = gaps_of(d)
        tr += [x for w, x in zip(d["word"].tolist(), g) if w in cid and np.isfinite(x) and x > 0]
    h = np.load(T.HERE.parents[2] / "data/raw/LibriBrain-Competition-2026/COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz")
    on = h["word_onsets_s"]; ho = np.diff(on, axis=1).ravel(); ho = ho[np.isfinite(ho)]
    q = [5, 25, 50, 75, 95]
    print("train gap pct", np.percentile(tr, q).round(3), "n", len(tr))
    print("holdout gap pct", np.percentile(ho, q).round(3), "n", len(ho), " <=0:", (ho <= 0).sum())
    for w in ["the", "a", "and", "of", "little", "nothing", "something"]:
        if w in cid: print(w, "llr by bin", llr[cid[w]].round(1))
    sys.exit()

if sys.argv[1] == "holdout":
    # evidence for the holdout sentence words (row order of the submission), fitted on all training runs
    edges, llr = table(False)
    h = np.load(T.HERE.parents[2] / "data/raw/LibriBrain-Competition-2026/COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz")
    G = []
    for on, m in zip(h["word_onsets_s"], h["word_mask"]):
        o = on[m.astype(bool)]; g = np.full(len(o), np.nan); g[:-1] = norm_gaps(np.diff(o)); G += g.tolist()
    E = ev(edges, llr, np.array(G))
    out = T.OUTS / f"dur_E_{NORM}.npy"; np.save(out, E)
    print("wrote", out, E.shape, "words with evidence", int((np.abs(E).sum(1) > 0).sum()))
    sys.exit()

tags = sys.argv[sys.argv.index("--tags") + 1:]
D = [np.load(T.OUTS / t / "proxy_logits.npz") for t in tags]
y, words = D[0]["y"], D[0]["words"].tolist()
lp = np.mean([d["ctx"] for d in D], 0)
# rebuild the proxy word order (same loop as proxy_eval.py) to get each word's gap
txt = Counter()
for f in files:
    if os.path.basename(f)[:-4] in VAL_RUNS: continue
    d = np.load(f, allow_pickle=True); w = d["word"].tolist()
    txt.update(" ".join(w[a:b]) for a, b in T.sentences(d["sentence"], d["onset"]))
gl, wl = [], []
for f in files:
    if os.path.basename(f)[:-4] not in VAL_RUNS: continue
    d = np.load(f, allow_pickle=True); w = d["word"].tolist(); g = gaps_of(d)
    for s, e in T.sentences(d["sentence"], d["onset"]):
        if txt[" ".join(w[s:e])] > 0: continue
        t = s
        while t < e:
            if w[t] not in cid: t += 1; continue
            u = t
            while u < e and w[u] in cid: u += 1
            if u - t >= 3:
                gl += g[t:u].tolist(); wl += w[t:u]
            t = u
assert wl == words, "proxy word order mismatch"
gl = np.array(gl)
edges, llr = table(True)
E = ev(edges, llr, gl)
np.save(T.OUTS / f"dur_E_proxy_{NORM}.npy", E)
print("proxy words", len(y), "with gap evidence", np.isfinite(gl).sum())
for tau in (0.0, 0.3):
    base = lp - tau * np.log(D[0]["prior"])
    print(f"tau{tau} " + "  ".join(f"b{b}:{bacc_topk(base + b * E, y):.4f}" for b in (0, 0.25, 0.5, 0.75, 1.0)))
print("duration only (no MEG):", round(bacc_topk(E, y), 4))
