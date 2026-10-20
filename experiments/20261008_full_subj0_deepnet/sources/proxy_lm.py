"""LM fusion on the holdout-like validation set (outputs/<tag>/proxy_logits.npz from proxy_eval.py).
Same fusion as the submission (lm_fusion.py soft PMI, lm_calib.py strength calibration), grid over tau and alpha.
  python proxy_lm.py --tags c7_l8 c5_dedup --lms gpt2 gpt2-medium
"""
import argparse
from pathlib import Path
import numpy as np
from lm_fusion import LM, fuse_sentence
from train import bacc_topk

OUTS = Path(__file__).resolve().parent.parent / "outputs"
ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--lms", nargs="+", default=["gpt2"])
ap.add_argument("--taus", nargs="+", type=float, default=[0.0, 0.3, 0.5, 0.7])
ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.3, 0.5, 0.7, 1.0])
ap.add_argument("--beam", type=int, default=32)
ap.add_argument("--device", default="cpu")
ap.add_argument("--dur", default="", help="duration evidence .npy (duration_prior.py proxy)")
ap.add_argument("--beta", type=float, default=0.5)
a = ap.parse_args()
D = [np.load(OUTS / t / "proxy_logits.npz") for t in a.tags]
y, seglen, prior = D[0]["y"], D[0]["seglen"], D[0]["prior"]
raw = np.mean([d["ctx"] for d in D], 0)
if a.dur:
    raw = raw + a.beta * np.load(a.dur)
lms = {name: LM(name, a.device) for name in a.lms}


def norm(x):
    x = x - x.max(1, keepdims=True); return x - np.log(np.exp(x).sum(1, keepdims=True))


for tau in a.taus:
    lp = norm(raw - tau * np.log(prior))
    pmis = []
    for name, lm in lms.items():
        mix, off = [], 0
        for n in seglen:
            mix.append(fuse_sentence(lm, lp[off:off + n], 0.5, a.beam, soft=True)); off += n
        mix = np.concatenate(mix)
        pmis.append(mix - np.log(np.exp(mix).mean(0)))
    ref = pmis[0].std()
    pmi = np.mean([p * (ref / p.std()) for p in pmis], 0)
    row = [f"alpha{al}: {bacc_topk(lp + al * pmi, y):.4f}" for al in a.alphas]
    print(f"tau{tau} " + "  ".join(row), flush=True)
