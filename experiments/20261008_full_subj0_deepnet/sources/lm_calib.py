"""Strength-calibrated LM ensemble for the sentence rows.

Each LM's evidence enters as PMI (its left-context predictive log-dist minus its average over holdout
positions; saved by lm_fusion.py --soft --pmi as *_lmmix.npy). Stronger LMs give sharper PMI, which acts
like a larger alpha. Here every LM's PMI is rescaled to the spread (std) of the reference LM (GPT-2) and the
rescaled PMIs are averaged, then fused with the MEG ensemble at the reference alpha.
  python lm_calib.py --tags ... --mix ref_lmmix.npy other_lmmix.npy --alpha 0.5 --out fused.csv
"""
import argparse, csv, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
COMP = HERE.parents[2]
def hl_path(t):
    """holdout logits of a tag; PNPL_RS=1 -> the input-rescaled ones (infer_rescale.py) when they exist"""
    d = HERE.parent / "outputs" / t
    return d / "holdout_logits_rs.npz" if os.environ.get("PNPL_RS") and (d / "holdout_logits_rs.npz").exists() else d / "holdout_logits.npz"


ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--mix", nargs="+", required=True, help="first = reference LM")
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--tau", type=float, default=0.0, help="logit adjustment of the MEG ensemble: - tau * log(prior)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
H = [np.load(hl_path(t)) for t in a.tags]
lp = np.mean([h["logp"] - a.tau * np.log(h["prior"]) for h in H], 0)
lp = lp - np.log(np.exp(lp - lp.max(1, keepdims=True)).sum(1, keepdims=True)) - lp.max(1, keepdims=True)
if os.environ.get("PNPL_DUR"):  # word-timing evidence for the sentence rows (duration_prior.py holdout)
    E = np.load(os.environ["PNPL_DUR"]); lp[:len(E)] += float(os.environ.get("PNPL_DUR_BETA", "0.5")) * E
    lp = lp - lp.max(1, keepdims=True); lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
pmis = []
for f in a.mix:
    mix = np.load(f)
    pmis.append(mix - np.log(np.exp(mix).mean(0)))
ref = pmis[0].std()
scaled = [p * (ref / p.std()) for p in pmis]
print("PMI std per LM:", [round(float(p.std()), 3) for p in pmis], "-> rescaled to", round(float(ref), 3))
pmi = np.mean(scaled, 0)
ns = len(pmi)
probs = np.exp(lp)
z = lp[:ns] + a.alpha * pmi
probs[:ns] = np.exp(z - z.max(1, keepdims=True))
probs /= probs.sum(1, keepdims=True)
header = open(COMP / "data/raw/deep_example_submission.csv", encoding="utf-8").readline().strip().split(",")[:51]
with open(a.out, "w", newline="", encoding="utf-8") as fo:
    w = csv.writer(fo); w.writerow(header)
    for i, r in enumerate(probs):
        w.writerow([i] + [f"{v:.6f}" for v in r])
print("wrote", a.out)
