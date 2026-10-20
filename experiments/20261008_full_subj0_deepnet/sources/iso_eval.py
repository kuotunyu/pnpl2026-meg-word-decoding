"""Isolated-word validation: every target word of the validation runs as a length-1 sequence.
For long-window data (50 Hz, onset at sample 25) the input is restricted to [onset, onset + 1 s] exactly like
the isolated-word holdout rows.
  python iso_eval.py <tag> [cpu|cuda] [data_dir_name]  -> outputs/<tag>/iso_val.npz and printed BAcc@10
"""
import glob, os, sys
import numpy as np, torch, torch.nn.functional as F
import train_ctx as T
from train import bacc_topk

tag = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
data = sys.argv[3] if len(sys.argv) > 3 else "subj0_pca_epochs"
DER = T.DER.parent / data
ck = torch.load(T.OUTS / tag / "model.pt", map_location=dev, weights_only=False)
cls = ck["classes"]; a = ck["args"]
m = T.CtxNet(len(cls), width=a.get("width", 320), depth=a.get("cdepth", 6), d=a.get("d", 512), layers=a["layers"],
             maxlen=max(a["maxlen"], 16)).to(dev)
m.load_state_dict(ck["state"]); m.prep = a.get("prep", ""); m.eval()
cid = {w: i for i, w in enumerate(cls[:50])}
O, Lc, Y = [], [], []
with torch.no_grad():
    for f in sorted(glob.glob(str(DER / "runs" / "*.npz"))):
        if os.path.basename(f)[:-4] not in T.VAL_RUNS:
            continue
        d = np.load(f, allow_pickle=True)
        y = np.array([cid.get(w, -1) for w in d["word"].tolist()]); k = y >= 0
        X = torch.from_numpy(d["X"][k]).float().clamp(-20, 20)
        if X.shape[-1] != 125:  # long-window data: keep only [onset, onset + 1 s]
            X[..., :T.ON] = 0; X[..., T.OFF:] = 0
        for i in range(0, len(X), 512):
            xb = X[i:i + 512, None].to(dev)
            lo, loc = m(xb, torch.zeros(len(xb), 1, dtype=torch.bool, device=dev))
            O.append(F.log_softmax(lo[:, 0, :50].float(), 1).cpu()); Lc.append(F.log_softmax(loc[:, 0, :50].float(), 1).cpu())
        Y.append(y[k])
O, Lc, Y = torch.cat(O).numpy(), torch.cat(Lc).numpy(), np.concatenate(Y)
prior = (ck["prior"][:50] / ck["prior"][:50].sum()).cpu().numpy()
np.savez(T.OUTS / tag / "iso_val.npz", out=O, local=Lc, y=Y, prior=prior)
for tau in (0.0, 0.5, 0.7, 1.0):
    adj = -tau * np.log(prior)[None]
    print(f"{tag} iso tau{tau}: ctx {bacc_topk(O + adj, Y):.4f} local {bacc_topk(Lc + adj, Y):.4f} "
          f"avg {bacc_topk((O + Lc) / 2 + adj, Y):.4f}", flush=True)
