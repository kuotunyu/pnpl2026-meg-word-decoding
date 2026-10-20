"""Compare ensembles of validation-split models on the two validations that Public cannot replace:
  proxy (holdout-like sentence runs, 1042 words; context head, prior correction tau 0.5)
  iso   (isolated-word validation, 7589 words; mean of context and context-free heads, tau 0.7 = Private iso rows)
Each ensemble = mean log-probs of its tags. Every ensemble is compared with the FIRST one by a paired bootstrap
(resampling proxy segments / iso words, 2000 draws): mean difference and the share of draws where it is > 0.
  python val_compare.py "base=r1_w3,r4_ema,r5_ep60" "arch=r1_w3,r4_ema,r5_ep60,rL_s1,rC_s1" ...
"""
import sys
import numpy as np
from pathlib import Path

OUTS = Path(__file__).resolve().parents[1] / "outputs"
K, NC = 10, 50


def hits(logits, y):
    return (np.argsort(-logits, 1)[:, :K] == y[:, None]).any(1)


def bacc(h, y, w=None):
    w = np.ones(len(y)) if w is None else w
    return float(np.mean([(h * w)[y == c].sum() / w[y == c].sum() for c in range(NC) if w[y == c].sum() > 0]))


def ens(tags, kind):
    ds = [np.load(OUTS / t / ("proxy_logits.npz" if kind == "proxy" else "iso_val.npz")) for t in tags]
    y, prior = ds[0]["y"], ds[0]["prior"]
    if kind == "proxy":
        lp = np.mean([d["ctx"] for d in ds], 0) - 0.5 * np.log(prior)
        return lp, y, ds[0]["seglen"]
    lp = np.mean([(d["out"] + d["local"]) / 2 for d in ds], 0) - 0.7 * np.log(prior)
    return lp, y, None


specs = [a.split("=", 1) for a in sys.argv[1:]]
rng = np.random.default_rng(0)
for kind in ("proxy", "iso"):
    res = {}
    for name, tl in specs:
        tags = tl.split(",")
        if not all((OUTS / t / ("proxy_logits.npz" if kind == "proxy" else "iso_val.npz")).exists() for t in tags):
            continue
        lp, y, seglen = ens(tags, kind)
        res[name] = (hits(lp, y), y, seglen)
    if not res:
        continue
    names = list(res); h0, y, seglen = res[names[0]]
    if kind == "proxy":
        seg = np.repeat(np.arange(len(seglen)), seglen); nunit = len(seglen)
    else:
        seg = np.arange(len(y)); nunit = len(y)
    draws = [np.bincount(seg, minlength=nunit)[0] * 0 + np.bincount(rng.integers(0, nunit, nunit), minlength=nunit)[seg]
             for _ in range(2000)]
    print(f"== {kind} ({len(y)} words) ==")
    for n in names:
        h = res[n][0]
        d = np.array([bacc(h, y, w) - bacc(h0, y, w) for w in draws]) if n != names[0] else np.zeros(1)
        print(f"{n:>14s} {bacc(h, y):.4f}  diff vs {names[0]} {d.mean():+.4f}  P(>0) {np.mean(d > 0):.2f}", flush=True)
