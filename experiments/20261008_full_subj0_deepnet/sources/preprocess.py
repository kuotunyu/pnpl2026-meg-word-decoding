"""Subject-0 MEG -> compact word epochs.

Per run: per-channel z-score over the whole run -> global PCA projection (306 -> K) ->
resample 250 Hz -> 125 Hz -> one [onset, onset+1 s] window (K x 125, float16) for EVERY word event.
The holdout files get the identical transform with statistics computed on the holdout file itself.

Usage:
  python preprocess.py pca        # fit global PCA basis from a sample of runs
  python preprocess.py runs [N]   # epoch all runs with N worker processes
  python preprocess.py holdout    # transform the two subj00 holdout files
"""
import glob, json, os, re, sys, time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

COMP = Path(__file__).resolve().parents[3]
RAW = COMP / "data" / "raw"
OUT = COMP / "data" / "derived" / "subj0_pca_epochs"
K = 80
WIN = 125  # samples at 125 Hz = 1 s

VOCAB = ["is", "the", "a", "to", "it", "i", "not", "was", "we", "be", "he", "that", "have", "this", "they", "of",
         "there", "and", "are", "in", "but", "will", "so", "all", "my", "for", "she", "were", "any", "really", "at",
         "out", "our", "am", "it's", "had", "him", "an", "very", "has", "do", "can", "time", "think", "good",
         "always", "new", "people", "as", "on"]
VID = {w: i for i, w in enumerate(VOCAB)}


def norm_word(w):
    w = str(w).strip().lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"^[^\w']+|[^\w']+$", "", w)


def runs():
    out = []
    for h5 in sorted(glob.glob(str(RAW / "LibriBrain*" / "*" / "derivatives" / "serialised" / "sub-0_*.h5"))):
        name = os.path.basename(h5).split("_proc-")[0]
        ev = Path(h5).parents[1] / "events" / f"{name}_events.tsv"
        if ev.exists():
            out.append((name, h5, str(ev)))
    return out


def zscore_inplace(x):
    m = x.mean(1, keepdims=True, dtype=np.float64).astype(np.float32)
    s = x.std(1, keepdims=True, dtype=np.float64).astype(np.float32)
    x -= m
    x /= np.maximum(s, 1e-20)
    return x


def fit_pca():
    rs = runs()
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rs), size=min(30, len(rs)), replace=False)
    cov = np.zeros((306, 306)); n = 0
    for i in pick:
        name, h5, _ = rs[i]
        with h5py.File(h5, "r") as f:
            x = f["data"][...][:, ::5].astype(np.float32)
        zscore_inplace(x)
        x = x.astype(np.float64)
        cov += x @ x.T; n += x.shape[1]
        print("pca", name, x.shape, flush=True)
    cov /= n
    ev, vec = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]; ev, vec = ev[order], vec[:, order]
    frac = np.cumsum(ev) / ev.sum()
    OUT.mkdir(parents=True, exist_ok=True)
    W = vec[:, :K].astype(np.float32)
    # scale components to unit variance so the network sees comparable inputs
    W = W / np.sqrt(ev[:K]).astype(np.float32)
    np.save(OUT / "pca_W.npy", W)
    json.dump({"K": K, "explained_at_K": float(frac[K - 1]), "eig": ev[:120].tolist(),
               "runs": [rs[i][0] for i in pick]}, open(OUT / "pca_info.json", "w"), indent=1)
    print("explained", frac[[39, 59, 69, 79, 99]], flush=True)


def transform(x, W):
    """x: (306, T) float32 already z-scored -> (K, T/2) float32 at 125 Hz."""
    y = W.T @ x
    return resample_poly(y, 1, 2, axis=1).astype(np.float32)


def do_run(args):
    name, h5, ev = args
    dst = OUT / "runs" / f"{name}.npz"
    if dst.exists():
        return name, "skip"
    t0 = time.time()
    W = np.load(OUT / "pca_W.npy")
    with h5py.File(h5, "r") as f:
        x = f["data"][...].astype(np.float32)
        t_first = float(f["times"][0])
    zscore_inplace(x)
    y = transform(x, W); del x
    df = pd.read_csv(ev, sep="\t")
    w = df[df["kind"].astype(str).str.lower() == "word"].dropna(subset=["segment", "timemeg"]).copy()
    w["norm"] = w["segment"].map(norm_word)
    on = w["timemeg"].astype(float).values
    start = (np.floor(on * 250).astype(np.int64)) // 2
    ok = (start >= 0) & (start + WIN <= y.shape[1])
    w, start = w[ok], start[ok]
    X = np.stack([y[:, s:s + WIN] for s in start]).astype(np.float16) if len(start) else np.zeros((0, K, WIN), np.float16)
    lab = np.array([VID.get(t, -1) for t in w["norm"]], np.int16)
    dur = pd.to_numeric(w.get("duration"), errors="coerce").fillna(-1).values.astype(np.float32)
    sent = pd.to_numeric(w.get("sentenceidx"), errors="coerce").fillna(-1).values.astype(np.int32)
    OUT.joinpath("runs").mkdir(parents=True, exist_ok=True)
    np.savez(dst, X=X, label=lab, word=w["norm"].values.astype(str), onset=on[ok].astype(np.float32),
             duration=dur, sentence=sent, t_first=t_first)
    return name, f"{len(lab)} words, {int((lab >= 0).sum())} vocab, {time.time() - t0:.0f}s"


def do_holdout():
    W = np.load(OUT / "pca_W.npy")
    hd = RAW / "LibriBrain-Competition-2026" / "COMPETITION_HOLDOUT"
    d = np.load(hd / "subj00_holdout01_sentence.npz")
    meg, nt, wm, on = d["meg"], d["sentence_n_times"], d["word_mask"], d["word_onsets_s"]
    cat = np.concatenate([meg[i][:, :nt[i]] for i in range(len(nt))], axis=1).astype(np.float64)
    m, s = cat.mean(1, keepdims=True), cat.std(1, keepdims=True)
    # rank / subspace check against the training PCA basis
    z = ((cat - m) / s)
    Q, _ = np.linalg.qr(W.astype(np.float64))
    zs = z[:, ::7]
    rel = np.linalg.norm(zs - Q @ (Q.T @ zs)) / np.linalg.norm(zs)
    Xs = []
    for i in range(len(nt)):
        x = ((meg[i][:, :nt[i]] - m) / s).astype(np.float32)
        y = transform(x, W)
        for j in np.nonzero(wm[i])[0]:
            st = int(round(float(on[i, j]) * 250)) // 2
            seg = y[:, st:st + WIN]
            if seg.shape[1] < WIN:
                seg = np.pad(seg, ((0, 0), (0, WIN - seg.shape[1])), mode="edge")
            Xs.append(seg)
    dw = np.load(hd / "subj00_holdout2_word.npz")["meg"].astype(np.float64)  # (92, 306, 250)
    cw = dw.transpose(1, 0, 2).reshape(306, -1)
    mw, sw = cw.mean(1, keepdims=True), cw.std(1, keepdims=True)
    Xw = [transform(((e - mw) / sw).astype(np.float32), W)[:, :WIN] for e in dw]
    X = np.stack(Xs + Xw).astype(np.float16)
    src = np.array([0] * len(Xs) + [1] * len(Xw), np.int8)
    np.savez(OUT / "holdout_subj00.npz", X=X, source=src)
    print("holdout", X.shape, "sentence", len(Xs), "word", len(Xw), "PCA-residual-frac", round(float(rel), 4))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pca":
        fit_pca()
    elif cmd == "runs":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        rs = runs(); print(len(rs), "runs", flush=True)
        with Pool(n) as p:
            for name, msg in p.imap_unordered(do_run, rs):
                print(name, msg, flush=True)
    elif cmd == "holdout":
        do_holdout()
