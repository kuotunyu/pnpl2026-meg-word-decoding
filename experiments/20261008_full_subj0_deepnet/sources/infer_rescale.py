"""Test-time input-scale fix. Training windows are z-scored with whole-run statistics; the holdout can only be
z-scored with its own short segments, and after PCA its windows come out ~1.35x larger (per-component std,
10/14 check). Rescale every holdout PCA component so its std over non-zero samples matches the training
windows (separately for sentence and isolated-word windows), then re-run inference for the given models.
  [PNPL_FS=50] python infer_rescale.py --data subj0_pca_w3 --tags w3f_s4 pw3_s1 ...
Writes outputs/<tag>/holdout_logits_rs.npz (same fields as holdout_logits.npz) and prints the scale factors.
"""
import argparse, glob
import numpy as np, torch
import train_ctx as T

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--nruns", type=int, default=24)
a = ap.parse_args()
DER = T.DER.parent / a.data
files = sorted(glob.glob(str(DER / "runs" / "*.npz")))
idx = np.linspace(0, len(files) - 1, min(a.nruns, len(files))).astype(int)


def comp_std(X):
    X = X.astype(np.float32); nz = X != 0
    n = nz.sum(axis=(0, 2)); s1 = (X * nz).sum(axis=(0, 2)); s2 = ((X ** 2) * nz).sum(axis=(0, 2))
    m = s1 / n; return np.sqrt(s2 / n - m ** 2)


tr = np.concatenate([np.load(files[i])["X"][::4][:600] for i in idx])
st = comp_std(tr)
h = np.load(DER / "holdout_subj00.npz"); X = h["X"].astype(np.float32); src = h["source"]
scale = np.ones((len(X), X.shape[1], 1), np.float32)
for s in (0, 1):
    r = st / comp_std(X[src == s]); scale[src == s] = r[None, :, None]
    print(f"source {s}: train/holdout std ratio mean {r.mean():.3f} min {r.min():.3f} max {r.max():.3f}", flush=True)
import os
if os.environ.get("NOSCALE"): scale[:] = 1.0
Xh = torch.from_numpy(X * scale).cuda()
hs, n = T.holdout_segments(); assert n == len(Xh)
Yh = torch.zeros(n, dtype=torch.long, device="cuda")
for tag in a.tags:
    if not (T.OUTS / tag / "model.pt").exists():
        print(f"{tag}: SKIP (no model.pt)", flush=True); continue
    ck = torch.load(T.OUTS / tag / "model.pt", map_location="cuda", weights_only=False); ia = ck["args"]
    m = T.CtxNet(len(ck["classes"]), width=ia.get("width", 320), depth=ia.get("cdepth", 6), d=ia.get("d", 512),
                 layers=ia["layers"], maxlen=max(ia["maxlen"], 16)).cuda()
    m.load_state_dict(ck["state"]); m.prep = ia.get("prep", "")
    P, Q, _ = T.run_eval(m, hs, Xh, Yh)
    pr = ck["prior"][:50]; pr = (pr / pr.sum()).cpu().numpy()
    np.savez(T.OUTS / tag / "holdout_logits_rs.npz", logp=P, local=Q, source=src, prior=pr)
    old = np.load(T.OUTS / tag / "holdout_logits.npz")["logp"] if (T.OUTS / tag / "holdout_logits.npz").exists() else None
    agree = (P.argmax(1) == old.argmax(1)).mean() if old is not None else float("nan")
    print(f"{tag}: top-1 agreement with unscaled inference {agree:.3f}", flush=True)
    del m; torch.cuda.empty_cache()
