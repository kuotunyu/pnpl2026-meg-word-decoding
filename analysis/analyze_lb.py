"""Public vs private analysis of all 99 scored submissions (data/submissions_raw.csv = Kaggle CLI export).
Writes data/analysis.json (every leaderboard statistic quoted in the paper), latex/fig_scatter.pdf and latex/fig_progression.pdf.
  python analyze_lb.py
"""
import csv, itertools, json, random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(H / "data/submissions_raw.csv", encoding="utf-8")))
S = {r["fileName"].removesuffix(".csv"): (float(r["publicScore"]), float(r["privateScore"]), r["date"]) for r in rows}
DEADLINE = "2026-10-16 12:00"
pre = {k: v for k, v in S.items() if v[2] < DEADLINE}          # submissions made before the deadline


def stage(k):
    if k.startswith("abl_"):
        return "cand"
    if any(t in k for t in ("SC0", "SCGL", "SCG2L", "AVG_SCGL")):
        return "cand"
    if S[k][2] >= "2026-10-14 16:00":                               # in-domain-LM submissions start with B5i_L1
        return "inlm"
    if S[k][0] < 0.5:
        return "early"
    return "meg"


from decimal import Decimal, ROUND_HALF_UP
RAW = {r["fileName"].removesuffix(".csv"): (Decimal(r["publicScore"]), Decimal(r["privateScore"])) for r in rows}
r3 = lambda d: float(Decimal(d).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))   # paper precision, from exact CSV values


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 2), round(float(c + h), 2)]


out = {"n_all": len(S), "n_pre_deadline": len(pre)}
for name, pool in (("pre", pre), ("all", S)):
    pub = np.array([v[0] for v in pool.values()]); prv = np.array([v[1] for v in pool.values()])
    for th in (0.0, 0.5, 0.60, 0.66, 0.69):
        m = pub >= th
        out[f"corr_{name}_pub_ge_{th}"] = [int(m.sum()), round(float(np.corrcoef(pub[m], prv[m])[0, 1]), 3)]
st = {k: stage(k) for k in S}
for g, members in (("early", ("early",)), ("meg", ("meg",)), ("inlm_only", ("inlm",)), ("cand", ("cand",)), ("language_side", ("inlm", "cand"))):
    ks = [k for k in pre if st[k] in members]
    out[f"stage_{g}"] = {"n": len(ks), "mean_pub": round(float(np.mean([S[k][0] for k in ks])), 4),
                         "mean_priv": round(float(np.mean([S[k][1] for k in ks])), 4),
                         "min_priv": min((S[k][1], k) for k in ks), "max_priv": max((S[k][1], k) for k in ks)}

# iso-only pairs: identical sentence rows, different isolated-word rows (checked on the submission files)
ISO_ONLY = [("finalW_calib2", "finalW_calib2_iso14"), ("final_B5", "final_B5i"), ("final_B9", "final_B9c"),
            ("sub_ef1c5c2_gs05", "sub_ef1c5c2_gs05_isoT1")]
out["iso_only_pairs"] = [[a, b, str(RAW[a][0]), str(RAW[b][0]), str(RAW[a][1]), str(RAW[b][1])] for a, b in ISO_ONLY]

# reference -> variant comparisons. kind: "single" = one documented change, "bundle" = several changes at once.
# val: what in-session validation suggested before the private release (component level); n/a = no matched result documented
PAIRS = [
    ("sentence-context model (redesign)", "sub_v1", "sub_c1", "n/a", "bundle"),
    ("closed-vocab LM fusion (PMI)", "sub_c5", "sub_c5_lmpmi05", "+", "single"),
    ("N-best hard marginals instead of soft PMI", "sub_c5_lmpmi05", "sub_c5_lm05", "n/a", "single"),
    ("Qwen3-8B instead of GPT-2", "final_gpt2", "final_qwen3_8b", "-", "single"),
    ("3 s window ensemble", "final_gpt2", "finalW_gpt2", "n/a", "bundle"),
    ("LM strength calibration (+ GPT-2-medium)", "finalW_gpt2", "finalW_calib2", "n/a", "single"),
    ("sentence prior correction tau0.5", "finalW_calib2", "finalW_calib2_t0.5", "+", "single"),
    ("higher 1 s share", "finalW_calib2_iso14", "finalY_calib2_t0", "+", "bundle"),
    ("self-training round 1", "finalW_calib2_iso14", "final_P1", "n/a", "single"),
    ("LM fusion after self-training", "final_P1noLM", "final_P1", "n/a", "single"),
    ("self-training round 2 (all replaced)", "final_P1", "final_P2", "n/a", "bundle"),
    ("sentence-end LM term", "final_P1", "final_P1eos", "n/a", "single"),
    ("larger diverse ensemble", "final_P1", "final_B2", "n/a", "bundle"),
    ("12-layer + wide-CNN members", "final_B2", "final_B5", "n/a", "bundle"),
    ("all members self-trained (one teacher)", "final_B2", "final_C", "n/a", "bundle"),
    ("more 1 s members", "final_B5", "final_B6a", "+", "bundle"),
    ("wide CNN instead of 12-layer", "final_B5", "final_F1", "+", "bundle"),
    ("sentence prior correction tau0.3", "final_B5i", "final_B5i_t0.3", "+", "single"),
    ("word-timing evidence beta0.5", "final_B5i", "final_B5d05", "+", "single"),
    ("word-timing evidence beta1.0", "final_B5i", "final_B5d10", "+", "single"),
    ("input rescaling", "final_B5i", "final_B5irs", "n/a", "single"),
    ("validation-motivated stack V", "final_B5i", "final_V", "+", "bundle"),
    ("lr-fixed deep members", "final_B5i", "final_B8b", "n/a", "bundle"),
    ("in-domain LM (807 sentences)", "final_B5i", "final_B5i_L1", "n/a", "single"),
    ("in-domain LM, 3-seed mixture", "final_B5i", "final_B5i_L1avg", "n/a", "bundle"),
    ("in-domain corpus 807 -> 1,368", "final_B5i_L1", "final_B5i_L3", "n/a", "single"),
    ("in-domain corpus 807 -> 2,103 (IN1)", "final_B5i_L1_IN1", "final_IN1_Q_llm_plus", "n/a", "single"),
    ("in-domain corpus 2,103 -> 3,742 (IN1)", "final_IN1_Q_llm_plus", "final_IN1_Q_big", "n/a", "single"),
    ("generic candidates, lambda0.15", "final_IN1_Q_llm_plus", "final_IN1Qp_SC0.15", "n/a", "single"),
    ("candidate decoding (guided pool + LM prior)", "final_IN1_Q_llm_plus", "final_IN1Qp_SCGL_b0.5", "n/a", "bundle"),
]
pairs = []
for name, a, b, val, kind in PAIRS:
    dp, dq = RAW[b][0] - RAW[a][0], RAW[b][1] - RAW[a][1]
    pairs.append({"change": name, "base": a, "variant": b, "d_pub": float(dp), "d_priv": float(dq),
                  "d_pub_3dp": r3(dp), "d_priv_3dp": r3(dq), "val": val, "kind": kind})
out["pairs"] = pairs
out["n_pairs"] = [len(pairs), sum(p["kind"] == "single" for p in pairs)]


def agree(sel, th):
    big = [p for p in sel if abs(p["d_pub"]) >= th]; small = [p for p in sel if abs(p["d_pub"]) < th]
    a = lambda L: int(sum(np.sign(p["d_pub"]) == np.sign(p["d_priv"]) for p in L))
    ds = np.array([[p["d_pub"], p["d_priv"]] for p in small])
    return {"big": [a(big), len(big), wilson(a(big), len(big))], "small": [a(small), len(small), wilson(a(small), len(small))],
            "corr_small": round(float(np.corrcoef(ds[:, 0], ds[:, 1])[0, 1]), 2)}


for th in (0.025, 0.03, 0.035):
    out[f"agree_all_th{th}"] = agree(pairs, th)
out["agree_single_th0.03"] = agree([p for p in pairs if p["kind"] == "single"], 0.03)
# contrasts where component-level validation and the public delta disagreed in sign
dis = [p for p in pairs if p["val"] in "+-" and np.sign(p["d_pub"]) != (1 if p["val"] == "+" else -1)]
gain = [p["d_priv"] if p["val"] == "+" else -p["d_priv"] for p in dis]   # private gain of following validation instead of public
out["disagreements"] = [[p["change"], p["d_pub_3dp"], p["d_priv_3dp"], "validation" if g > 0 else "public"] for p, g in zip(dis, gain)]
nr = int(sum(g > 0 for g in gain))
out["disagree_summary"] = {"n": len(dis), "validation_right": nr, "wilson": wilson(nr, len(dis)),
                           "mean_gain": round(float(np.mean(gain)), 4), "max_gain": round(float(max(gain)), 4),
                           "max_loss": round(float(-min(gain)), 4)}

# final-selection back-test (submissions made before the deadline only)
ACTUAL = ["final_AVG_SCGL_Qbig", "final_IN1Qp_SCGL_b0.5", "final_IN1_Q_big", "final_IN1_Q_llm_plus", "final_B9cIN_Qplus",
          "final_B5i_L1_IN1", "final_IN1Qp_SC0.15", "final_B5i_L1", "final_B9c_L1", "final_IN1_Q_big_SCG2L_l0.5"]
EARLIER = ["final_IN1_Q_llm_plus", "final_B5i_L1_IN1", "final_B5i_L1", "final_B9c_L1", "final_B5i", "final_B9c",
           "final_M1", "final_B5d10", "final_V", "final_F1"]          # selection recorded on 10/15 before the final reselection
best = lambda ks: max((S[k][1], k) for k in ks)
top10 = sorted(pre, key=lambda k: -pre[k][0])[:10]
out["select_actual"] = best(ACTUAL)
out["select_top10_public"] = best(top10)
out["select_top10_vs_actual_overlap"] = len(set(top10) & set(ACTUAL))
out["select_earlier"] = best(EARLIER)
pre_ld = [k for k in pre if st[k] not in ("inlm", "cand") and pre[k][0] >= 0.6]
out["select_top10_public_no_late_lm"] = best(sorted(pre_ld, key=lambda k: -pre[k][0])[:10])
cand_pool = [k for k in pre if pre[k][0] >= 0.66]
random.seed(0)
draws = np.array([best(random.sample(cand_pool, 10))[0] for _ in range(20000)])
out["select_random10_pub_ge_0.66"] = {"pool": len(cand_pool), "mean": round(float(draws.mean()), 4),
                                      "p10": round(float(np.percentile(draws, 10)), 4), "p90": round(float(np.percentile(draws, 90)), 4),
                                      "frac_le_actual": round(float((draws <= out["select_actual"][0]).mean()), 4)}
out["best_private_overall"] = sorted(((v[1], k, v[0]) for k, v in pre.items()), reverse=True)[:5]
out["note_ranks"] = ("public 13/90 (Kaggle CLI 2026-10-20); private 44/91 at 0.66304; V's 0.68767 falls between ranks 33 (0.68780) "
                     "and 34 (0.68580) (private leaderboard page 2026-10-25)")
out["repeated_public_examples"] = {"0.72001": [k for k in pre if RAW[k][0] == Decimal("0.72001")],
                                   "0.71301": [k for k in pre if RAW[k][0] == Decimal("0.71301")]}
out["ablations_post_deadline"] = {k: [str(RAW[k][0]), str(RAW[k][1])] for k in S if k.startswith("abl_")}
json.dump(out, open(H / "data/analysis.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1, ensure_ascii=False))
for p in pairs:
    print(f"{p['change'][:44]:44s} {p['d_pub_3dp']:+.3f} {p['d_priv_3dp']:+.3f} val:{p['val']:3s} {p['kind']}")

# Figure: public vs private scatter
col = {"early": "#9E9E9E", "meg": "#4C72B0", "inlm": "#DD8452", "cand": "#C44E52"}
lab = {"early": "per-word / early", "meg": "MEG-side development", "inlm": "in-domain LM", "cand": "candidate decoding"}
fig, ax = plt.subplots(figsize=(5.5, 3.4))
for g in ("early", "meg", "inlm", "cand"):
    ks = [k for k in S if st[k] == g and S[k][0] >= 0.5]
    ax.scatter([S[k][0] for k in ks], [S[k][1] for k in ks], s=14, c=col[g], label=lab[g], alpha=0.85, edgecolors="none")
ax.scatter([S[k][0] for k in ACTUAL], [S[k][1] for k in ACTUAL], s=48, facecolors="none", edgecolors="k", linewidths=0.8,
           label="final selection")
for k, t, off in (("final_V", "V", (5, 2)), ("final_gpt2", "best private", (-52, 3)), ("final_IN1Qp_SCGL_b0.5", "best public", (4, 4)),
                  ("final_B5i_L1", "in-domain LM", (4, 4))):
    ax.annotate(t, (S[k][0], S[k][1]), xytext=off, textcoords="offset points", fontsize=7)
lim = (0.52, 0.745)
ax.plot(lim, lim, ls=":", c="0.6", lw=0.8)
ax.set_xlim(lim); ax.set_ylim(0.52, 0.705)
ax.set_xlabel("Public BAcc@10", fontsize=8); ax.set_ylabel("Private BAcc@10", fontsize=8)
ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, loc="lower right", frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(H / "fig_scatter.png", dpi=150)

# Figure: milestones, public and private
M = [("Linear baseline", "submission_deep_final"), ("Per-word CNN", "sub_v1"), ("Sentence Transformer", "sub_c1"),
     ("+ dedup", "sub_c5"), ("+ closed-vocab LM (PMI)", "sub_c5_lmpmi05"), ("9-model ensemble (best private)", "final_gpt2"),
     ("3 s windows, LM calibration", "finalW_calib2"),
     ("+ self-training", "final_P1"), ("+ diverse ensemble (B5)", "final_B5"), ("+ in-domain LM (807 sent.)", "final_B5i_L1"),
     ("IN1 + LM on 3,742 sent.", "final_IN1_Q_big"), ("+ candidate decoding", "final_IN1Qp_SCGL_b0.5")]
fig, ax = plt.subplots(figsize=(5.5, 2.6))
x = np.arange(len(M))
ax.plot(x, [S[k][0] for _, k in M], "o-", c="#4C72B0", ms=3.5, lw=1.2, label="public")
ax.plot(x, [S[k][1] for _, k in M], "s-", c="#C44E52", ms=3.5, lw=1.2, label="private")
ax.set_xticks(x); ax.set_xticklabels([m for m, _ in M], rotation=35, ha="right", fontsize=6.5)
ax.set_ylabel("BAcc@10", fontsize=8); ax.tick_params(axis="y", labelsize=7); ax.legend(fontsize=7, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(H / "fig_progression.png", dpi=150)
print("figures written")
