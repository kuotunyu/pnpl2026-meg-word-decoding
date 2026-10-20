"""LM predictive mixtures on the holdout-like validation set, for offline LM comparison.
Same soft left-context fusion as lm_fusion.py, applied to the proxy segments of proxy_eval.py
(MEG = mean log-probs of the given tags' proxy_logits.npz). Saves outputs/proxy_mix_<name>.npy.
  python proxy_mix.py --tags r1_w3 r1_c7ref --lm Qwen/Qwen3-8B --name qwen3_8b --device cuda
"""
import argparse
from pathlib import Path
import numpy as np
from lm_fusion import LM, fuse_sentence

OUTS = Path(__file__).resolve().parent.parent / "outputs"
ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--lm", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--beam", type=int, default=32)
a = ap.parse_args()
D = [np.load(OUTS / t / "proxy_logits.npz") for t in a.tags]
lp = np.mean([d["ctx"] for d in D], 0)
lp = lp - lp.max(1, keepdims=True); lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
lm = LM(a.lm, a.device)
mix, off = [], 0
for n in D[0]["seglen"]:
    mix.append(fuse_sentence(lm, lp[off:off + n], 0.5, a.beam, soft=True)); off += n
np.save(OUTS / f"proxy_mix_{a.name}.npy", np.concatenate(mix))
print("wrote", OUTS / f"proxy_mix_{a.name}.npy")
