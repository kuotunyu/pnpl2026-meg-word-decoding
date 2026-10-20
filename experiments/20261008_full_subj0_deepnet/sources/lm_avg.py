"""Average several fused submissions (geometric mean of probabilities) -> one fused CSV.
  python lm_avg.py --inputs a.csv b.csv --out c.csv"""
import argparse, csv
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--inputs", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
dfs = [pd.read_csv(f) for f in a.inputs]
lp = np.mean([np.log(np.clip(d.values[:, 1:].astype(float), 1e-9, None)) for d in dfs], 0)
p = np.exp(lp - lp.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(list(dfs[0].columns))
    for i, r in enumerate(p):
        w.writerow([i] + [f"{v:.6f}" for v in r])
print("wrote", a.out, p.shape)
