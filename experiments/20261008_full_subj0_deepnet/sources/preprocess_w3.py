"""Long-window variant of preprocess.py: every word gets [onset-0.5 s, onset+2.5 s] at 50 Hz (80 x 150),
zero-filled where the recording has no samples. Same per-run z-score and the SAME PCA basis (pca_W.npy of
subj0_pca_epochs). Holdout windows are zero-filled exactly where the holdout provides no MEG (before a
sentence's first word, after the sentence end, outside [0, 1 s] for isolated words).

  python preprocess_w3.py runs 8
  python preprocess_w3.py holdout
"""
import sys, time
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from preprocess import OUT as BASE, RAW, VID, norm_word, runs, zscore_inplace

import os
POST = float(os.environ.get("PNPL_POST", "2.5"))  # seconds after onset (2.5 -> 3 s window "w3", 3.5 -> "w4")
FS = int(os.environ.get("PNPL_FS", "50"))  # samples per second (50 -> "w3"; 100 -> "w3h")
PRE = FS // 2  # 0.5 s before onset
WIN = PRE + int(round(POST * FS))
OUT = BASE.parent / (f"subj0_pca_w{int(round(0.5 + POST))}" + ("" if FS == 50 else "h"))


def project(x, W):
    from math import gcd
    g = gcd(FS, 250)
    return resample_poly(W.T @ x, FS // g, 250 // g, axis=1).astype(np.float32)


def cut(y, start):
    out = np.zeros((y.shape[0], WIN), np.float32)
    a, b = max(start, 0), min(start + WIN, y.shape[1])
    if b > a:
        out[:, a - start:b - start] = y[:, a:b]
    return out


def do_run(args):
    name, h5, ev = args
    dst = OUT / "runs" / f"{name}.npz"
    if dst.exists():
        return name, "skip"
    t0 = time.time()
    W = np.load(BASE / "pca_W.npy")
    with h5py.File(h5, "r") as f:
        x = f["data"][...].astype(np.float32)
    zscore_inplace(x)
    y = project(x, W); del x
    df = pd.read_csv(ev, sep="\t")
    w = df[df["kind"].astype(str).str.lower() == "word"].dropna(subset=["segment", "timemeg"]).copy()
    w["norm"] = w["segment"].map(norm_word)
    on = w["timemeg"].astype(float).values
    # keep exactly the words the 1 s pipeline keeps, so indices/labels align with subj0_pca_epochs
    start125 = (np.floor(on * 250).astype(np.int64)) // 2
    T125 = (y.shape[1] * 5) // 2
    ok = (start125 >= 0) & (start125 + 125 <= T125)
    w, on = w[ok], on[ok]
    X = np.stack([cut(y, int(round(o * FS)) - PRE) for o in on]).astype(np.float16)
    lab = np.array([VID.get(t, -1) for t in w["norm"]], np.int16)
    sent = pd.to_numeric(w.get("sentenceidx"), errors="coerce").fillna(-1).values.astype(np.int32)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    np.savez(dst, X=X, label=lab, word=w["norm"].values.astype(str), onset=on.astype(np.float32), sentence=sent)
    return name, f"{len(lab)} words {time.time() - t0:.0f}s"


def do_holdout():
    W = np.load(BASE / "pca_W.npy")
    hd = RAW / "LibriBrain-Competition-2026" / "COMPETITION_HOLDOUT"
    d = np.load(hd / "subj00_holdout01_sentence.npz")
    meg, nt, wm, on = d["meg"], d["sentence_n_times"], d["word_mask"], d["word_onsets_s"]
    cat = np.concatenate([meg[i][:, :nt[i]] for i in range(len(nt))], axis=1).astype(np.float64)
    m, s = cat.mean(1, keepdims=True), cat.std(1, keepdims=True)
    Xs = []
    for i in range(len(nt)):
        y = project(((meg[i][:, :nt[i]] - m) / s).astype(np.float32), W)
        for j in np.nonzero(wm[i])[0]:
            Xs.append(cut(y, int(round(float(on[i, j]) * FS)) - PRE))
    dw = np.load(hd / "subj00_holdout2_word.npz")["meg"].astype(np.float64)
    cw = dw.transpose(1, 0, 2).reshape(306, -1)
    mw, sw = cw.mean(1, keepdims=True), cw.std(1, keepdims=True)
    Xw = []
    for e in dw:
        y = project(((e - mw) / sw).astype(np.float32), W)  # 50 samples = [0, 1 s]
        z = np.zeros((y.shape[0], WIN), np.float32); z[:, PRE:PRE + y.shape[1]] = y
        Xw.append(z)
    X = np.stack(Xs + Xw).astype(np.float16)
    src = np.array([0] * len(Xs) + [1] * len(Xw), np.int8)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "holdout_subj00.npz", X=X, source=src)
    print("holdout", X.shape, "sentence", len(Xs), "word", len(Xw))


if __name__ == "__main__":
    if sys.argv[1] == "runs":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        rs = runs(); print(len(rs), "runs", flush=True)
        with Pool(n) as p:
            for name, msg in p.imap_unordered(do_run, rs):
                print(name, msg, flush=True)
    else:
        do_holdout()
