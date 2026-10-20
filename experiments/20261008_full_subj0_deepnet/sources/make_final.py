"""Assemble a submission: sentence rows from an LM-fusion CSV, isolated-word rows from the MEG ensemble
(mean of each model's context head and context-free head, logit-adjusted by tau).
  python make_final.py --fused ../outputs/sub_x.csv --tags f_s1 c5_dedup --tau 0.7 --out ../outputs/final_x.csv
"""
import argparse, csv, os
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
COMP = HERE.parents[2]
def hl_path(t):
    """holdout logits of a tag; PNPL_RS=1 -> the input-rescaled ones (infer_rescale.py) when they exist"""
    d = HERE.parent / "outputs" / t
    return d / "holdout_logits_rs.npz" if os.environ.get("PNPL_RS") and (d / "holdout_logits_rs.npz").exists() else d / "holdout_logits.npz"


ap = argparse.ArgumentParser()
ap.add_argument("--fused", required=True)
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--tau", type=float, default=0.7)
ap.add_argument("--iso_tags", nargs="+", default=None, help="models for isolated-word rows (default: --tags)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
base = pd.read_csv(a.fused).values[:, 1:].astype(float)
H = [np.load(hl_path(t)) for t in (a.iso_tags or a.tags)]
src = H[0]["source"]; iso = src == 1
lp = np.mean([(h["logp"] + h["local"]) / 2 - a.tau * np.log(h["prior"]) for h in H], 0)
p = np.exp(lp - lp.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
out = base.copy(); out[iso] = p[iso]
header = open(COMP / "data/raw/deep_example_submission.csv", encoding="utf-8").readline().strip().split(",")[:51]
with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(header)
    for i, r in enumerate(out):
        w.writerow([i] + [f"{v:.6f}" for v in r])
print("wrote", a.out, "isolated rows", int(iso.sum()))
