"""Public vs private analysis of all 99 scored submissions (data/submissions_raw.csv = Kaggle CLI export).
Writes data/analysis.json (every number quoted in the write-up), fig_scatter.png and fig_progression.png.
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


out = {"n_all": len(S), "n_pre_deadline": len(pre)}
pub = np.array([v[0] for v in S.values()]); prv = np.array([v[1] for v in S.values()])
for th in (0.0, 0.60, 0.66, 0.69):
    m = pub >= th
    out[f"corr_pub_ge_{th}"] = [int(m.sum()), round(float(np.corrcoef(pub[m], prv[m])[0, 1]), 3)]
st = {k: stage(k) for k in S}
for g in ("early", "meg", "inlm", "cand"):
    ks = [k for k in S if st[k] == g]
    out[f"stage_{g}"] = {"n": len(ks), "mean_pub": round(float(np.mean([S[k][0] for k in ks])), 4),
                         "mean_priv": round(float(np.mean([S[k][1] for k in ks])), 4),
                         "best_priv": max((S[k][1], k) for k in ks)}

# paired comparisons: (change, base, variant, what in-session validation predicted before the private release)
PAIRS = [
    ("sentence-context model", "sub_v1", "sub_c1", "n/a"),
    ("closed-vocab LM fusion (PMI)", "sub_c5", "sub_c5_lmpmi05", "+"),
    ("N-best hard marginals instead of soft PMI", "sub_c5_lmpmi05", "sub_c5_lm05", "n/a"),
    ("Qwen3-8B instead of GPT-2", "final_gpt2", "final_qwen3_8b", "-"),
    ("3 s windows + LM calibration", "final_gpt2", "finalW_calib2", "n/a"),
    ("sentence prior correction tau0.5", "finalW_calib2", "finalW_calib2_t0.5", "+"),
    ("higher 1 s share", "finalW_calib2_iso14", "finalY_calib2_t0", "+"),
    ("self-training round 1", "finalW_calib2_iso14", "final_P1", "n/a"),
    ("LM fusion after self-training", "final_P1noLM", "final_P1", "n/a"),
    ("self-training round 2 (all replaced)", "final_P1", "final_P2", "n/a"),
    ("sentence-end LM term", "final_P1", "final_P1eos", "n/a"),
    ("larger diverse ensemble", "final_P1", "final_B2", "n/a"),
    ("12-layer + wide-CNN members", "final_B2", "final_B5", "n/a"),
    ("all members self-trained (one teacher)", "final_B2", "final_C", "n/a"),
    ("more 1 s members", "final_B5", "final_B6a", "+"),
    ("wide CNN instead of 12-layer", "final_B5", "final_F1", "+"),
    ("sentence prior correction tau0.3", "final_B5i", "final_B5i_t0.3", "+"),
    ("word-timing evidence", "final_B5i", "final_B5d10", "+"),
    ("input rescaling", "final_B5i", "final_B5irs", "n/a"),
    ("validation-optimal stack V", "final_B5i", "final_V", "+"),
    ("lr-fixed deep members", "final_B5i", "final_B8b", "n/a"),
    ("in-domain LM (807 sentences)", "final_B5i", "final_B5i_L1", "n/a"),
    ("in-domain LM, 3-seed mixture", "final_B5i", "final_B5i_L1avg", "n/a"),
    ("in-domain LM on 3,742 sentences (IN1)", "final_B5i_L1_IN1", "final_IN1_Q_big", "n/a"),
    ("candidate decoding (guided + LM prior)", "final_IN1_Q_llm_plus", "final_IN1Qp_SCGL_b0.5", "n/a"),
]
pairs = []
for name, a, b, val in PAIRS:
    dp, dq = S[b][0] - S[a][0], S[b][1] - S[a][1]
    pairs.append({"change": name, "base": a, "variant": b, "d_pub": round(dp, 4), "d_priv": round(dq, 4), "val": val})
out["pairs"] = pairs
dp = np.array([p["d_pub"] for p in pairs]); dq = np.array([p["d_priv"] for p in pairs])
big = np.abs(dp) >= 0.03; small = ~big
out["pairs_sign_agree_big"] = [int((np.sign(dp[big]) == np.sign(dq[big])).sum()), int(big.sum())]
out["pairs_sign_agree_small"] = [int((np.sign(dp[small]) == np.sign(dq[small])).sum()), int(small.sum())]
out["pairs_corr_small"] = round(float(np.corrcoef(dp[small], dq[small])[0, 1]), 3)
# decisions on which validation and public disagreed in sign: who matched private?
dis = [p for p in pairs if p["val"] in "+-" and np.sign(p["d_pub"]) != (1 if p["val"] == "+" else -1)]
out["val_vs_pub_disagreements"] = [[p["change"], p["d_pub"], p["d_priv"],
                                     "validation" if np.sign(p["d_priv"]) == (1 if p["val"] == "+" else -1) else "public"] for p in dis]

# final-selection strategies (submissions made before the deadline only)
ACTUAL = ["final_AVG_SCGL_Qbig", "final_IN1Qp_SCGL_b0.5", "final_IN1_Q_big", "final_IN1_Q_llm_plus", "final_B9cIN_Qplus",
          "final_B5i_L1_IN1", "final_IN1Qp_SC0.15", "final_B5i_L1", "final_B9c_L1", "final_IN1_Q_big_SCG2L_l0.5"]
EARLIER = ["final_IN1_Q_llm_plus", "final_B5i_L1_IN1", "final_B5i_L1", "final_B9c_L1", "final_B5i", "final_B9c",
           "final_M1", "final_B5d10", "final_V", "final_F1"]          # selection on 10/15 before the final reselection
best = lambda ks: max((S[k][1], k) for k in ks)
top10 = sorted(pre, key=lambda k: -pre[k][0])[:10]
out["select_actual"] = best(ACTUAL)
out["select_earlier"] = best(EARLIER)
out["select_top10_public"] = best(top10)
pre_ld = [k for k in pre if st[k] != "inlm" and st[k] != "cand" and pre[k][0] >= 0.6]
out["select_top10_public_no_late_lm"] = best(sorted(pre_ld, key=lambda k: -pre[k][0])[:10])
cand_pool = [k for k in pre if pre[k][0] >= 0.66]
random.seed(0)
draws = [best(random.sample(cand_pool, 10))[0] for _ in range(20000)]
out["select_random10_pub_ge_0.66"] = {"pool": len(cand_pool), "mean": round(float(np.mean(draws)), 4),
                                      "p10": round(float(np.percentile(draws, 10)), 4), "p90": round(float(np.percentile(draws, 90)), 4)}
out["best_private_overall"] = sorted(((v[1], k, v[0]) for k, v in pre.items()), reverse=True)[:5]
out["note_private_lb"] = "rank 44/91 at 0.66304 (private leaderboard page, 2026-10-25)"

# repeated public values (granularity of the public split)
vals = [v[0] for v in pre.values()]
from collections import Counter
rep = Counter(vals)
out["public_values_repeated"] = sorted([[v, c] for v, c in rep.items() if c >= 3], key=lambda x: -x[1])[:6]
out["public_distinct"] = [len(rep), len(vals)]
json.dump(out, open(H / "data/analysis.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1, ensure_ascii=False))
for p in pairs:
    print(f"{p['change'][:42]:42s} {p['d_pub']:+.3f} {p['d_priv']:+.3f} val:{p['val']}")

# Figure: public vs private scatter
col = {"early": "#9E9E9E", "meg": "#4C72B0", "inlm": "#DD8452", "cand": "#C44E52"}
lab = {"early": "per-word / early", "meg": "MEG-side development", "inlm": "in-domain LM", "cand": "candidate decoding"}
fig, ax = plt.subplots(figsize=(5.5, 3.4))
for g in ("early", "meg", "inlm", "cand"):
    ks = [k for k in S if st[k] == g and S[k][0] >= 0.5]
    ax.scatter([S[k][0] for k in ks], [S[k][1] for k in ks], s=14, c=col[g], label=lab[g], alpha=0.85, edgecolors="none")
ax.scatter([S[k][0] for k in ACTUAL], [S[k][1] for k in ACTUAL], s=48, facecolors="none", edgecolors="k", linewidths=0.8,
           label="final selection")
for k, t in (("final_V", "V"), ("final_IN1Qp_SCGL_b0.5", "best public"), ("final_B5i_L1", "in-domain LM")):
    ax.annotate(t, (S[k][0], S[k][1]), xytext=(4, 4), textcoords="offset points", fontsize=7)
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
     ("+ dedup", "sub_c5"), ("+ closed-vocab LM (PMI)", "sub_c5_lmpmi05"), ("+ 3 s windows, LM calibration", "finalW_calib2"),
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
