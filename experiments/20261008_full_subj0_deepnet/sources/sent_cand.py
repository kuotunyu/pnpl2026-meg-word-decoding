"""Whole-sentence candidate decoding for the holdout sentence rows.

The holdout sentences are short conversational utterances over the 50 words, and each sentence's length is
known (word_mask). For a sentence of n words, every corpus sentence with n words is a candidate; its score is
the summed MEG ensemble log-prob of its words at each position (all positions at once, so later words inform
earlier ones). Candidate posteriors (temperature T) give per-position word marginals, which are mixed into an
existing submission's sentence rows with weight lam.
  python sent_cand.py --tags ... --corpus ../outputs/vocab_corpus_llm_big.txt --base ../outputs/final_x.csv \
      --T 1.0 --lam 0.3 --out ../outputs/final_x_sc.csv
"""
import argparse, csv
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from lm_fusion import VOCAB, hl_path, LM

HERE = Path(__file__).resolve().parent
COMP = HERE.parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--corpus", nargs="+", required=True)
ap.add_argument("--base", required=True, help="submission whose sentence rows are mixed with the candidate marginals")
ap.add_argument("--T", type=float, default=1.0)
ap.add_argument("--lam", type=float, default=0.3)
ap.add_argument("--tau", type=float, default=0.0)
ap.add_argument("--smooth", type=float, default=0.02, help="uniform mass inside the candidate marginals")
ap.add_argument("--geo", action="store_true", help="geometric mix (log-linear) instead of arithmetic")
ap.add_argument("--lm", default=None, help="score candidate fluency with this LM (sentence prior)")
ap.add_argument("--beta", type=float, default=0.5)
ap.add_argument("--meg_w", type=float, default=1.0, help="weight of the MEG term in the candidate score (0 = language-only control)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

idx = {w: i for i, w in enumerate(VOCAB)}
sents = []
for f in a.corpus:
    for l in Path(f).read_text(encoding="utf-8").splitlines():
        ws = l.strip().lower().split()
        if ws and all(w in idx for w in ws):
            sents.append(tuple(ws))
sents = list(dict.fromkeys(sents))
bylen = {}
for s in sents:
    bylen.setdefault(len(s), []).append([idx[w] for w in s])
bylen = {n: np.array(v) for n, v in bylen.items()}
prior = {n: np.zeros(len(v)) for n, v in bylen.items()}
if a.lm:
    lm = LM(a.lm, "cuda")
    for n, v in bylen.items():
        prior[n] = lm.seq_loglik(v.tolist())
    print("LM sentence log-lik per length (median):", {n: round(float(np.median(p)), 1) for n, p in prior.items()})
print("candidates", len(sents), "by length", sorted((n, len(v)) for n, v in bylen.items()))

H = [np.load(hl_path(t)) for t in a.tags]
lp = np.mean([h["logp"] - a.tau * np.log(h["prior"]) for h in H], 0)
lp = lp - lp.max(1, keepdims=True); lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
try:
    wm = np.load(COMP / "data/raw/LibriBrain-Competition-2026/COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz")["word_mask"]
except Exception:
    wm = np.loadtxt(HERE / "holdout_wordmask.csv", delimiter=",").astype(bool)
lens = wm.sum(1).astype(int)
print("holdout sentence lengths", sorted(Counter(lens.tolist()).items()))

base = pd.read_csv(a.base).values[:, 1:].astype(float)
out = base.copy()
off, conf, top1 = 0, [], []
for n in lens:
    seg = lp[off:off + n]
    C = bylen.get(n)
    if C is None or len(C) < 20:
        off += n; continue
    sc = (a.meg_w * seg[np.arange(n)[None, :], C].sum(1) + a.beta * prior[n]) / a.T
    q = np.exp(sc - sc.max()); q /= q.sum()
    m = np.full((n, 50), a.smooth / 50)
    for j in range(n):
        m[j] += (1 - a.smooth) * np.bincount(C[:, j], weights=q, minlength=50)
    b = base[off:off + n]
    if a.geo:
        z = (1 - a.lam) * np.log(b + 1e-9) + a.lam * np.log(m)
        r = np.exp(z - z.max(1, keepdims=True))
    else:
        r = (1 - a.lam) * b + a.lam * m
    out[off:off + n] = r / r.sum(1, keepdims=True)
    conf.append(q.max()); top1.append(" ".join(VOCAB[i] for i in C[q.argmax()]))
    off += n
print("sentence rows", off, "| median top-candidate posterior", round(float(np.median(conf)), 3),
      "| >0.5:", int((np.array(conf) > 0.5).sum()), "of", len(conf))
print("examples:", top1[:8])
chg = (out[:off].argmax(1) != base[:off].argmax(1)).mean()
print("top-1 changed on sentence rows:", round(float(chg), 3))
header = open(COMP / "data/raw/deep_example_submission.csv", encoding="utf-8").readline().strip().split(",")[:51]
with open(a.out, "w", newline="", encoding="utf-8") as fo:
    w = csv.writer(fo); w.writerow(header)
    for i, r in enumerate(out):
        w.writerow([i] + [f"{v:.6f}" for v in r])
print("wrote", a.out)
