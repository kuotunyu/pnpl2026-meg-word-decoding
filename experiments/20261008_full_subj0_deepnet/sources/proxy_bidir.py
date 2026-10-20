"""LM fusion variants on the holdout-like validation (proxy) segments, scored with top-10 balanced accuracy:
  left  = current submission fusion (soft left-context PMI, lm_fusion.fuse_sentence soft=True)
  bidir = whole-sentence LM evidence for each position given the other positions from the top-K joint
          hypotheses (fuse_sentence bidir=K), as PMI
  both  = left + bidir PMI averaged (each rescaled to the left PMI's std)
MEG = mean ctx log-probs of the tags' proxy_logits.npz with prior correction tau.
  python proxy_bidir.py --tags r4_ema r5_ep60 r5_h100 r1_c7ref --lm gpt2 --device cuda --k 8
"""
import argparse, time
from pathlib import Path
import numpy as np
from lm_fusion import LM, fuse_sentence
from train import bacc_topk

OUTS = Path(__file__).resolve().parent.parent / "outputs"
ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--lm", default="gpt2")
ap.add_argument("--device", default="cuda")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--beam", type=int, default=32)
ap.add_argument("--tau", type=float, default=0.0)
a = ap.parse_args()
D = [np.load(OUTS / t / "proxy_logits.npz") for t in a.tags]
y, seglen, prior = D[0]["y"], D[0]["seglen"], D[0]["prior"]


def norm(x):
    m = x.max(1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


lp = norm(np.mean([d["ctx"] for d in D], 0) - a.tau * np.log(prior))
lm = LM(a.lm, a.device)
t0 = time.time(); L_, B_, off = [], [], 0
for n in seglen:
    seg = lp[off:off + n]
    L_.append(fuse_sentence(lm, seg, 0.5, a.beam, soft=True))
    B_.append(fuse_sentence(lm, seg, 0.5, a.beam, soft=True, bidir=a.k)); off += n
left, bid = np.concatenate(L_), np.concatenate(B_)
np.save(OUTS / f"proxy_bidir_{a.lm.replace('/', '_')}_k{a.k}.npy", np.stack([left, bid]))
pl = left - np.log(np.exp(left).mean(0)); pb = bid - np.log(np.exp(bid).mean(0))
pb_s = pb * (pl.std() / pb.std())
print(f"tags {a.tags} lm {a.lm} k {a.k} tau {a.tau}  ({time.time() - t0:.0f}s)  PMI std left {pl.std():.3f} bidir {pb.std():.3f}")
print(f"MEG only {bacc_topk(lp, y):.4f}")
for name, p in [("left", pl), ("bidir", pb), ("bidir_scaled", pb_s), ("both", (pl + pb_s) / 2)]:
    print(f"{name:>12s} " + "  ".join(f"a{al}:{bacc_topk(lp + al * p, y):.4f}" for al in (0.2, 0.3, 0.5, 0.7, 1.0)), flush=True)
