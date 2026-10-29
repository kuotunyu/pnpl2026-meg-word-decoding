"""Compare submission pairs that differ only in the 92 isolated-word rows (rows 868-959).
  python iso_check.py <folder with the submission csv files>  ->  data/iso_check.json
Reports, per pair, the largest difference on the 868 sentence rows and how many isolated rows have a different
top-10 set; the scores come from data/submissions_raw.csv.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
SUBS = Path(sys.argv[1])
N_SENT = 868
PAIRS = [("finalW_calib2", "finalW_calib2_iso14"), ("final_B5", "final_B5i"), ("final_B9", "final_B9c"),
         ("sub_ef1c5c2_gs05", "sub_ef1c5c2_gs05_isoT1")]
RAW = {r["fileName"][:-4]: (r["publicScore"], r["privateScore"]) for r in csv.DictReader(open(H / "data/submissions_raw.csv", encoding="utf-8"))}


def load(name):
    p = SUBS / f"{name}.csv"
    if not p.exists():
        return None
    return np.loadtxt(p, delimiter=",", skiprows=1, usecols=range(1, 51), encoding="utf-8")


def top10(x):
    return [frozenset(r) for r in np.argsort(-x, 1)[:, :10]]


out = []
for a, b in PAIRS:
    rec = {"reference": a, "variant": b, "public": [RAW[a][0], RAW[b][0]], "private": [RAW[a][1], RAW[b][1]]}
    A, B = load(a), load(b)
    if A is None or B is None:
        rec["files"] = "not retained"
    else:
        rec["sentence_rows_max_abs_diff"] = float(np.abs(A[:N_SENT] - B[:N_SENT]).max())
        ta, tb = top10(A[N_SENT:]), top10(B[N_SENT:])
        rec["isolated_rows"] = len(ta)
        rec["isolated_rows_top10_differs"] = int(sum(x != y for x, y in zip(ta, tb)))
        rec["isolated_rows_top1_differs"] = int((A[N_SENT:].argmax(1) != B[N_SENT:].argmax(1)).sum())
    out.append(rec)
    print(rec)
json.dump(out, open(H / "data/iso_check.json", "w", encoding="utf-8"), indent=1)
