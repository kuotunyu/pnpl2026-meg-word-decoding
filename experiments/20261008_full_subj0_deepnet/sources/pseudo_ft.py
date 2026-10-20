"""Self-training on the unlabeled holdout: fine-tune a full-data model with the holdout recordings as extra
training data, using the current best submission's probabilities as soft targets (no labels, no metadata).
Each step mixes a normal training batch (hard labels, as in train_ctx.py) with a batch of holdout segments
(whole sentences, or single isolated words) whose target is the teacher distribution over the 50 words.
  python pseudo_ft.py --init w3f_s1 --teacher teacher.csv --data subj0_pca_w3 --w3aug --tag pw3_s1 [--fs 50]
Writes outputs/<tag>/model.pt and holdout_logits.npz (same format as train_ctx.py).
"""
import argparse, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import train_ctx as T

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True)
ap.add_argument("--teacher", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--data", default="subj0_pca_w3")
ap.add_argument("--w3aug", action="store_true")
ap.add_argument("--epochs", type=float, default=3.0, help="passes over the training segments")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--hw", type=float, default=1.0, help="weight of the holdout soft-target loss")
ap.add_argument("--hbs", type=int, default=32, help="holdout segments per step")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
T.DER = T.DER.parent / a.data
ck = torch.load(T.OUTS / a.init / "model.pt", map_location="cuda", weights_only=False)
ia = ck["args"]
t0 = time.time()
X, Y, tr, _, _, classes = T.load(ia.get("full", True), ia["big"], ia["maxlen"], not ia.get("nodedup", False))  # validation-split init keeps its val runs out
assert classes == ck["classes"]
NC = len(classes)
T.ISVOC = ((Y >= 0) & (Y < 50)).cpu().numpy()
prior = ck["prior"].cuda(); w = prior ** (-ia["wpow"]); w = w / (w * prior).sum()
model = T.CtxNet(NC, width=ia.get("width", 320), depth=ia.get("cdepth", 6), d=ia.get("d", 512), layers=ia["layers"],
                 maxlen=max(ia["maxlen"], 16)).cuda()
model.load_state_dict(ck["state"]); model.prep = ia.get("prep", "")
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999), use_buffers=True)

h = np.load(T.DER / "holdout_subj00.npz")
Xh = torch.from_numpy(h["X"]).cuda()
hs, n = T.holdout_segments(); assert n == len(Xh)
teach = pd.read_csv(a.teacher).values[:, 1:].astype(np.float32)
assert teach.shape == (n, 50)
Th = torch.from_numpy(teach / teach.sum(1, keepdims=True)).cuda()
print(f"train segs {len(tr)} holdout segs {len(hs)} load {time.time() - t0:.0f}s", flush=True)

bs = ia["bs"]; nb = int(len(tr) // bs * a.epochs)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=ia["wd"])
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=nb, pct_start=0.05)
g = torch.Generator().manual_seed(a.seed)
model.train(); tl = th = 0.0
for step in range(nb):
    segs = [tr[j] for j in torch.randint(0, len(tr), (bs,), generator=g).tolist()]
    xb, yb, pad = T.batch_from(segs, X, Y, True, g)
    hsel = [hs[j] for j in torch.randint(0, len(hs), (a.hbs,), generator=g).tolist()]
    L = max(e - s for s, e in hsel)
    I = torch.zeros((len(hsel), L), dtype=torch.long); hp = torch.ones((len(hsel), L), dtype=torch.bool)
    for r, (s, e) in enumerate(hsel):
        I[r, :e - s] = torch.arange(s, e); hp[r, :e - s] = False
    I, hp = I.cuda(), hp.cuda()
    xh = Xh[I].float().clamp(-20, 20)
    xh = xh * (1 + 0.1 * torch.randn(xh.shape[0], xh.shape[1], 1, 1, device=xh.device)) + 0.1 * torch.randn_like(xh) * (xh != 0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo, loc = model(T.augment(xb, g, a.w3aug, False), pad)
        lh, lhl = model(xh, hp)
    yf = yb.reshape(-1)
    loss = F.cross_entropy(lo.float().reshape(-1, NC), yf, weight=w, label_smoothing=0.1, ignore_index=-100) \
        + 0.5 * F.cross_entropy(loc.float().reshape(-1, NC), yf, weight=w, label_smoothing=0.1, ignore_index=-100)
    m = ~hp
    tgt = Th[I][m]
    lsoft = -(tgt * F.log_softmax(lh.float()[m][:, :50], 1)).sum(1).mean() \
        - 0.5 * (tgt * F.log_softmax(lhl.float()[m][:, :50], 1)).sum(1).mean()
    tot = loss + a.hw * lsoft
    opt.zero_grad(set_to_none=True); tot.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); ema.update_parameters(model)
    tl += loss.item(); th += lsoft.item()
    if (step + 1) % 500 == 0 or step == nb - 1:
        print(json.dumps({"step": step + 1, "of": nb, "loss": round(tl / 500, 4), "hold": round(th / 500, 4),
                          "sec": round(time.time() - t0)}), flush=True); tl = th = 0.0
model = ema.module
out = T.OUTS / a.tag; out.mkdir(parents=True, exist_ok=True)
torch.save({"state": model.state_dict(), "args": {**ia, "pseudo_from": a.init, "teacher": a.teacher}, "classes": classes,
            "prior": ck["prior"]}, out / "model.pt")
del X
P, Q, _ = T.run_eval(model, hs, Xh, torch.zeros(n, dtype=torch.long, device="cuda"))
pr = ck["prior"][:50]; pr = (pr / pr.sum()).cpu().numpy()
np.savez(out / "holdout_logits.npz", logp=P, local=Q, source=h["source"], prior=pr)
agree = (P.argmax(1) == np.log(teach + 1e-9).argmax(1)).mean()
print("DONE", a.tag, "teacher top-1 agreement", round(float(agree), 3), round(time.time() - t0), "s", flush=True)
