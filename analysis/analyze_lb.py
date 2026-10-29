"""Public vs private analysis of all 99 scored submissions (data/submissions_raw.csv = Kaggle CLI export).
Writes data/analysis.json (every leaderboard statistic quoted in the paper) and the figures ../paper/fig_*.pdf (previews in data/*.png).
Hand-coded inputs (pair choices, labels, selections, validation evidence) are listed below with their sources.
  python analyze_lb.py
"""
import csv, json, math, random
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from itertools import product
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.5})   # embedded TrueType, no Type 3

H = Path(__file__).resolve().parent
OUT = H.parent / "paper"     # figure output folder
rows = list(csv.DictReader(open(H / "data/submissions_raw.csv", encoding="utf-8")))
S = {r["fileName"].removesuffix(".csv"): (float(r["publicScore"]), float(r["privateScore"]), r["date"]) for r in rows}
RAW = {r["fileName"].removesuffix(".csv"): (Decimal(r["publicScore"]), Decimal(r["privateScore"])) for r in rows}
assert len(S) == len(rows), "duplicate file names in the export"
DEADLINE = "2026-10-16 12:00"                                    # UTC, Kaggle deadline
pre = {k: v for k, v in S.items() if v[2] < DEADLINE}
r3 = lambda d: float(Decimal(d).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def stage(k):
    """early: public < 0.5; late: on or after the first in-domain-LM submission (2026-10-14 16:00 UTC); meg: the rest."""
    if S[k][0] < 0.5:
        return "early"
    if S[k][2] >= "2026-10-14 16:00":
        return "late"
    return "meg"


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 2), round(c + h, 2)]


def fisher_ci(r, n):
    z, se = math.atanh(r), 1 / math.sqrt(n - 3)
    return [round(math.tanh(z - 1.96 * se), 2), round(math.tanh(z + 1.96 * se), 2)]


corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
out = {"n_all": len(S), "n_pre_deadline": len(pre), "post_deadline": sorted(k for k in S if k not in pre)}

# --- correlations --------------------------------------------------------------------------------------------
pub = np.array([v[0] for v in pre.values()]); prv = np.array([v[1] for v in pre.values()])
for th in (0.0, 0.5, 0.6, 0.66, 0.69):
    m = pub >= th
    r = corr(pub[m], prv[m])
    out[f"corr_pre_ge_{th}"] = [int(m.sum()), round(r, 3), fisher_ci(r, int(m.sum()))]
st = {k: stage(k) for k in S}
fam = {}
arr = {}
for g in ("meg", "late"):
    ks = [k for k in pre if st[k] == g and pre[k][0] >= 0.66]
    a = np.array([pre[k][0] for k in ks]); b = np.array([pre[k][1] for k in ks]); arr[g] = (a, b)
    fam[g] = {"n": len(ks), "mean_pub": round(float(a.mean()), 4), "mean_priv": round(float(b.mean()), 4), "r": round(corr(a, b), 2),
              "r_ci": fisher_ci(corr(a, b), len(ks)), "sd_pub": round(float(a.std(ddof=1)), 4), "sd_priv": round(float(b.std(ddof=1)), 4),
              "first": min(S[k][2] for k in ks)[:16], "last": max(S[k][2] for k in ks)[:16]}
out["family_pub_ge_0.66"] = fam
welch = lambda i: math.sqrt(arr["late"][i].var(ddof=1) / len(arr["late"][i]) + arr["meg"][i].var(ddof=1) / len(arr["meg"][i]))
out["family_contrast"] = {"d_pub": round(float(arr["late"][0].mean() - arr["meg"][0].mean()), 4), "se_pub": round(welch(0), 4),
                          "d_priv": round(float(arr["late"][1].mean() - arr["meg"][1].mean()), 4), "se_priv": round(welch(1), 4)}  # nominal
for g in ("early", "meg", "late"):
    ks = [k for k in pre if st[k] == g]
    out[f"stage_{g}"] = {"n": len(ks), "mean_pub": round(float(np.mean([S[k][0] for k in ks])), 4),
                         "mean_priv": round(float(np.mean([S[k][1] for k in ks])), 4)}
inlm = [k for k in pre if st[k] == "late" and not any(t in k for t in ("SC0", "SCGL", "SCG2L", "AVG_SCGL"))]
out["inlm_private_range"] = [min((S[k][1], k) for k in inlm), max((S[k][1], k) for k in inlm), len(inlm)]
out["qwen_only_vs_plus"] = {"qwen_1575": [S["final_IN1_Q_llm"][0], S["final_IN1_Q_llm"][1]],
                            "qwen_plus_claude_2103": [S["final_IN1_Q_llm_plus"][0], S["final_IN1_Q_llm_plus"][1]],
                            "d_priv": float(RAW["final_IN1_Q_llm_plus"][1] - RAW["final_IN1_Q_llm"][1])}
L1FAM = ["final_B5i_L1", "final_B5i_L1s2", "final_B5i_L1_IN1", "final_B5i_L1_IN2", "final_B9c_L1"]   # 807-sentence LM, public >= 0.718
out["l1_family"] = {"public": [min(S[k][0] for k in L1FAM), max(S[k][0] for k in L1FAM)],
                    "private": [min(S[k][1] for k in L1FAM), max(S[k][1] for k in L1FAM)]}
out["distinct_values_pre"] = {"public": len(set(RAW[k][0] for k in pre)), "private": len(set(RAW[k][1] for k in pre)), "n": len(pre)}
out["repeated_public_0.72001"] = sorted(k for k in pre if RAW[k][0] == Decimal("0.72001"))

# --- seed repeats of one recipe: B5i + GPT-2 fine-tuned on 807 sentences + GPT-2-medium; only the fine-tuning seed of the
#     in-domain GPT-2 differs, the MEG ensemble is the same (REPORT 10/15 02:50) ------------------------------------------
SEEDS = ["final_B5i_L1", "final_B5i_L1s1", "final_B5i_L1s2"]
sp = np.array([S[k][1] for k in SEEDS]); sq = np.array([S[k][0] for k in SEEDS])
sd = float(sp.std(ddof=1))
floor = 2 * math.sqrt(2) * sd                       # twice the sd of a difference of two independent draws: a rough scale
d_ref = sp - S["final_B5i"][1]
out["seed_repeats"] = {"private": [S[k][1] for k in SEEDS], "public": [S[k][0] for k in SEEDS],
                       "sd_private": round(sd, 4), "sd_public": round(float(sq.std(ddof=1)), 4),
                       "sd_private_ci95": [round(sd * math.sqrt(2 / 7.3778), 4), round(sd * math.sqrt(2 / 0.050636), 4)],  # chi2, 2 df
                       "mean_private": round(float(sp.mean()), 4), "mixture_private": S["final_B5i_L1avg"][1],
                       "noise_floor_2sd_diff": round(floor, 3),
                       "vs_B5i": {"d_priv": [round(float(x), 5) for x in d_ref], "mean": round(float(d_ref.mean()), 4),
                                  "se": round(sd / math.sqrt(len(SEEDS)), 4)}}

# --- isolated-word-only pairs: identical sentence rows, different isolated rows (checked on the submission files) --
ISO_ONLY = [("finalW_calib2", "finalW_calib2_iso14"), ("final_B5", "final_B5i"), ("final_B9", "final_B9c"),
            ("sub_ef1c5c2_gs05", "sub_ef1c5c2_gs05_isoT1")]
out["iso_only_pairs"] = [[a, b, str(RAW[a][0]), str(RAW[b][0]), str(RAW[a][1]), str(RAW[b][1])] for a, b in ISO_ONLY]

# --- reference -> variant comparisons -------------------------------------------------------------------------
# kind: "single" = one documented change, "bundle" = several changes at once.
# val: direction suggested by in-session validation (novel-sentence or proxy) before the private release, coded only when the
#      documented difference exceeded the proxy noise (about 0.015); "n/a" otherwise. evidence: REPORT.md entry (Taiwan time).
PAIRS = [
    ("sentence-context model (redesign)", "sub_v1", "sub_c1", "n/a", "bundle", ""),
    ("closed-vocab LM fusion (PMI)", "sub_c5", "sub_c5_lmpmi05", "+", "single", "10/11 16:50 proxy 0.533 -> 0.555"),
    ("LM fusion without PMI (N-best)", "sub_c5_lmpmi05", "sub_c5_lm05", "n/a", "bundle", ""),
    ("Qwen3-8B instead of GPT-2", "final_gpt2", "final_qwen3_8b", "-", "single", "10/11 16:50 proxy Qwen3-8B 0.52 vs GPT-2 0.555"),
    ("3 s window ensemble", "final_gpt2", "finalW_gpt2", "n/a", "bundle", ""),
    ("GPT-2-medium added, strength-calibrated", "finalW_gpt2", "finalW_calib2", "n/a", "bundle", ""),
    ("sentence prior correction tau0.5", "finalW_calib2", "finalW_calib2_t0.5", "+", "single", "10/09 20:35 novel-sentence 0.668 -> 0.718"),
    ("higher 1 s share", "finalW_calib2_iso14", "finalY_calib2_t0", "n/a", "bundle",
     "10/09 21:40 proxy 0.525 -> 0.561 also adds four models; 10/09 20:35 novel-sentence 0.668 -> 0.672 (27% -> 50% 1 s)"),
    ("self-training round 1", "finalW_calib2_iso14", "final_P1", "n/a", "single", ""),
    ("LM fusion after self-training", "final_P1noLM", "final_P1", "n/a", "single", ""),
    ("self-training round 2 (all replaced)", "final_P1", "final_P2", "n/a", "bundle", ""),
    ("sentence-end LM term", "final_P1", "final_P1eos", "n/a", "single", ""),
    ("larger diverse ensemble", "final_P1", "final_B2", "n/a", "bundle", ""),
    ("12-layer + wide-CNN members", "final_B2", "final_B5", "n/a", "bundle", ""),
    ("all members self-trained (one teacher)", "final_B2", "final_C", "n/a", "bundle", ""),
    ("more 1 s members", "final_B5", "final_B6a", "n/a", "bundle", "conflicting: proxy favoured more 1 s, novel-sentence did not (10/11 16:00)"),
    ("wide CNN instead of 12-layer", "final_B5", "final_F1", "+", "bundle", "10/13 11:20 proxy +0.020 for the wide CNN"),
    ("sentence prior correction tau0.3", "final_B5i", "final_B5i_t0.3", "+", "single", "10/09 21:00 proxy 0.558 -> 0.592"),
    ("word-timing evidence beta0.5", "final_B5i", "final_B5d05", "+", "single", "10/13 20:40 proxy 0.533 -> 0.603"),
    ("word-timing evidence beta1.0", "final_B5i", "final_B5d10", "+", "single", "10/13 20:40 proxy 0.533 -> 0.613"),
    ("input rescaling", "final_B5i", "final_B5irs", "n/a", "single", ""),
    ("validation-motivated stack V", "final_B5i", "final_V", "+", "bundle", "10/14 13:30, built from the tau0.3 and timing results"),
    ("12-layer members, lower learning rate", "final_B5i", "final_B8b", "n/a", "bundle", ""),
    ("in-domain LM (807 sentences, seed 0)", "final_B5i", "final_B5i_L1", "n/a", "single", ""),
    ("in-domain LM, 3-seed mixture", "final_B5i", "final_B5i_L1avg", "n/a", "bundle", ""),
    ("in-domain corpus 807 -> 1,368", "final_B5i_L1", "final_B5i_L3", "n/a", "single", ""),
    ("in-domain corpus 807 -> 2,103 (+ Qwen sentences; IN1)", "final_B5i_L1_IN1", "final_IN1_Q_llm_plus", "n/a", "single", ""),
    ("in-domain corpus 2,103 -> 3,742 (IN1)", "final_IN1_Q_llm_plus", "final_IN1_Q_big", "n/a", "single", ""),
    ("generic candidates, lambda0.15", "final_IN1_Q_llm_plus", "final_IN1Qp_SC0.15", "n/a", "single", ""),
    ("candidate decoding (guided pool + LM prior)", "final_IN1_Q_llm_plus", "final_IN1Qp_SCGL_b0.5", "n/a", "bundle", ""),
]
EARLY_STRUCTURAL = 4                                   # first four pairs: structural changes made on 7-9 October
order = sorted(pre, key=lambda k: S[k][2])
best_so_far = {}
m = -1
for k in order:
    m = max(m, S[k][0]); best_so_far[k] = m
pairs = []
for name, a, b, val, kind, ev in PAIRS:
    dp, dq = RAW[b][0] - RAW[a][0], RAW[b][1] - RAW[a][1]
    pairs.append({"change": name, "base": a, "variant": b, "d_pub": float(dp), "d_priv": float(dq), "d_pub_3dp": r3(dp),
                  "d_priv_3dp": r3(dq), "val": val, "kind": kind, "evidence": ev, "ref_was_public_max": S[a][0] >= best_so_far[a]})
out["pairs"] = pairs
out["n_pairs"] = [len(pairs), sum(p["kind"] == "single" for p in pairs), sum(p["kind"] == "bundle" for p in pairs)]
out["refs_running_public_max"] = [sum(p["ref_was_public_max"] for p in pairs), len(pairs)]


def agree(sel, th=None):
    a = lambda L: int(sum(np.sign(p["d_pub"]) == np.sign(p["d_priv"]) for p in L))
    if th is None:
        return [a(sel), len(sel), wilson(a(sel), len(sel))]
    big = [p for p in sel if abs(p["d_pub"]) >= th]; small = [p for p in sel if abs(p["d_pub"]) < th]
    return {"big": [a(big), len(big), wilson(a(big), len(big))], "small": [a(small), len(small), wilson(a(small), len(small))]}


out["agree_early_structural"] = agree(pairs[:EARLY_STRUCTURAL])
later = pairs[EARLY_STRUCTURAL:]
out["agree_later"] = agree(later)
lp = np.array([[p["d_pub"], p["d_priv"]] for p in later])
rl = corr(lp[:, 0], lp[:, 1]); out["corr_later"] = [round(rl, 2), fisher_ci(rl, len(later))]
out["later_mean_deltas"] = [round(float(lp[:, 0].mean()), 4), round(float(lp[:, 1].mean()), 4)]
out["later_large_public"] = [[p["change"], p["d_pub_3dp"], p["d_priv_3dp"]] for p in later if abs(p["d_pub"]) >= 0.03]
for th in (0.025, 0.03, 0.035):
    out[f"agree_all_th{th}"] = agree(pairs, th)
out["n_small_priv_effects"] = [sum(abs(p["d_priv"]) < floor for p in pairs), len(pairs)]
out["later_floor"] = {"n": len(later), "priv_below": int(sum(abs(p["d_priv"]) < floor for p in later)),
                      "pub_above": int(sum(abs(p["d_pub"]) >= floor for p in later)),
                      "both_above": int(sum(abs(p["d_pub"]) >= floor and abs(p["d_priv"]) >= floor for p in later)),
                      "large_public_priv_below": [int(sum(abs(p["d_priv"]) < floor for p in later if abs(p["d_pub"]) >= 0.03)),
                                                  len(out["later_large_public"])],
                      "sd_d_pub": round(float(lp[:, 0].std(ddof=1)), 4), "sd_d_priv": round(float(lp[:, 1].std(ddof=1)), 4)}
flips = [p for p in pairs if np.sign(p["d_pub"]) != np.sign(p["d_priv"])]
out["sign_flips_below_floor"] = [int(sum(abs(p["d_priv"]) < floor for p in flips)), len(flips)]

# --- validation vs public ------------------------------------------------------------------------------------
dis = [p for p in pairs if p["val"] in "+-" and np.sign(p["d_pub"]) != (1 if p["val"] == "+" else -1)]
gain = np.array([p["d_priv"] if p["val"] == "+" else -p["d_priv"] for p in dis])


def signrank_p(x):
    """exact two-sided Wilcoxon signed-rank p-value (no ties in |x| here)."""
    r = np.argsort(np.argsort(np.abs(x))) + 1
    w = r[x > 0].sum(); tot = r.sum(); n = len(x)
    dist = [sum(rr for rr, s in zip(range(1, n + 1), sg) if s) for sg in product([0, 1], repeat=n)]
    lo = min(w, tot - w)
    return round(2 * sum(d <= lo for d in dist) / len(dist), 2)


tq = {5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}[len(gain) - 1]
se = float(gain.std(ddof=1) / math.sqrt(len(gain)))
out["disagreements"] = [[p["change"], p["d_pub_3dp"], p["d_priv_3dp"], "validation" if g > 0 else "public", p["evidence"]]
                        for p, g in zip(dis, gain)]
out["disagree_summary"] = {"n": len(dis), "validation_right": int((gain > 0).sum()), "mean_gain": round(float(gain.mean()), 4),
                           "ci95": [round(float(gain.mean()) - tq * se, 3), round(float(gain.mean()) + tq * se, 3)],
                           "signrank_p": signrank_p(gain), "max_gain": round(float(gain.max()), 4),
                           "max_loss": round(float(-gain.min()), 4), "below_floor": int((np.abs(gain) < floor).sum()),
                           "max_abs_gain_5dp": round(float(np.abs(gain).max()), 5), "floor_5dp": round(floor, 5)}

# --- final-selection back-test (pre-deadline submissions only) ------------------------------------------------
ACTUAL = ["final_AVG_SCGL_Qbig", "final_IN1Qp_SCGL_b0.5", "final_IN1_Q_big", "final_IN1_Q_llm_plus", "final_B9cIN_Qplus",
          "final_B5i_L1_IN1", "final_IN1Qp_SC0.15", "final_B5i_L1", "final_B9c_L1", "final_IN1_Q_big_SCG2L_l0.5"]   # REPORT 10/16 13:45
EARLIER = ["final_IN1_Q_llm_plus", "final_B5i_L1_IN1", "final_B5i_L1", "final_B9c_L1", "final_B5i", "final_B9c",
           "final_M1", "final_B5d10", "final_V", "final_F1"]                                                        # REPORT 10/15 11:45
out["selections"] = {"actual": ACTUAL, "earlier": EARLIER, "replaced_on_last_day": sorted(set(EARLIER) - set(ACTUAL))}
best = lambda ks: max((S[k][1], k) for k in ks)
top10 = sorted(pre, key=lambda k: -pre[k][0])[:10]
out["select_actual"] = best(ACTUAL)
out["select_top10_public"] = best(top10)
out["select_top10_vs_actual_overlap"] = len(set(top10) & set(ACTUAL))
out["select_earlier"] = best(EARLIER)
out["select_top10_public_not_late"] = best(sorted([k for k in pre if st[k] == "meg"], key=lambda k: -pre[k][0])[:10])


def exact_best_of_10(pool, x):
    """P(best private of 10 random distinct members <= x) and E[best], exactly."""
    v = sorted(pre[k][1] for k in pool); n = len(v)
    p_le = math.comb(sum(s <= x for s in v), 10) / math.comb(n, 10)
    e = sum(v[i] * (math.comb(i + 1, 10) - math.comb(i, 10)) for i in range(n)) / math.comb(n, 10)
    return {"pool": n, "p_le_actual": round(p_le, 5), "expected_best": round(e, 4), "pool_max": max(v)}


x0 = out["select_actual"][0]
pool = [k for k in pre if pre[k][0] >= 0.66]
out["backtest_pool_ge_0.66"] = exact_best_of_10(pool, x0)
out["backtest_late_family"] = exact_best_of_10([k for k in pool if st[k] == "late"], x0)
out["backtest_late_family"]["n_above_actual"] = int(sum(pre[k][1] > x0 for k in pool if st[k] == "late"))
out["backtest_threshold_sensitivity"] = {th: exact_best_of_10([k for k in pre if pre[k][0] >= th], x0) for th in (0.64, 0.65, 0.66, 0.67, 0.68)}
random.seed(0)
draws = np.array([best(random.sample(pool, 10))[0] for _ in range(20000)])
out["backtest_mc"] = {"draws": len(draws), "mean": round(float(draws.mean()), 4), "p10": round(float(np.percentile(draws, 10)), 4),
                      "p90": round(float(np.percentile(draws, 90)), 4), "frac_le_actual": round(float((draws <= x0).mean()), 5)}
out["best_private_overall"] = sorted(((v[1], k, v[0]) for k, v in S.items()), reverse=True)[:3]

# --- ranks from saved leaderboard snapshots -------------------------------------------------------------------
priv_lb = [float(r["score"]) for r in csv.DictReader(open(H / "data/private_leaderboard_top49_2026-10-29.csv", encoding="utf-8"))]
pub_lb = list(csv.DictReader(open(H / "data/public_leaderboard_2026-10-29.csv", encoding="utf-8")))
rank_of = lambda x: 1 + sum(s > x for s in priv_lb)
out["ranks"] = {"private_actual": rank_of(x0), "private_if_V": rank_of(out["select_earlier"][0]),
                "public": int([r["rank"] for r in pub_lb if r["team"] == "yu_kuo14"][0]),
                "teams": sum(int(r["rank"]) > 0 for r in pub_lb),                  # rank 0 = the host's unranked baseline row
                "unranked_rows": sum(int(r["rank"]) == 0 for r in pub_lb)}

json.dump(out, open(H / "data/analysis.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k not in ("pairs", "selections")}, indent=1, ensure_ascii=False))
for p in pairs:
    print(f"{p['change'][:46]:46s} {p['d_pub_3dp']:+.3f} {p['d_priv_3dp']:+.3f} val:{p['val']:3s} {p['kind']:6s} refmax:{p['ref_was_public_max']}")

# ================================== figures (sized to the 5.5-inch text width) ==================================
for s_ in ("axes.spines.top", "axes.spines.right"):
    plt.rcParams[s_] = False

# Figure 1: system overview
fig, ax = plt.subplots(figsize=(5.5, 1.32)); ax.set_xlim(-0.12, 10.12); ax.set_ylim(0, 2.25); ax.axis("off")
B = [(0.0, "MEG word\nepochs", "#E8EEF7", "-"), (1.7, "CNN word\nencoder", "#E8EEF7", "-"),
     (3.4, "sentence\nTransformer\n(~30 models)", "#E8EEF7", "-"), (5.1, "closed-vocab\nLM fusion", "#FBEEE6", "-"),
     (6.8, "candidate\ndecoding", "#FBE3E3", "--"), (8.5, "50-way\nprobabilities", "#F2F2F2", "-")]
w, h, y = 1.5, 1.2, 0.55
for x, t, c, ls in B:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", fc=c, ec="0.35", lw=0.8, ls=ls))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5)
for (x, *_), (x2, *__) in zip(B[:-1], B[1:]):
    ax.annotate("", xy=(x2 - 0.02, y + h / 2), xytext=(x + w + 0.02, y + h / 2), arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
ax.text(3.0, 0.18, "self-training on unlabelled test MEG", ha="center", fontsize=7, color="0.3", style="italic")
ax.text(5.85, 2.05, "GPT-2 + GPT-2-medium; in-domain GPT-2 added in the last two days", ha="center", fontsize=7, color="#B5651D")
ax.text(7.55, 0.18, "added in the last two days", ha="center", fontsize=7, color="#C44E52", style="italic")
fig.tight_layout(pad=0.1); fig.savefig(OUT / "fig_system.pdf"); fig.savefig(H / "data/fig_system.png", dpi=130)

# Figure 2: (a) public vs private of the 95 pre-deadline submissions, (b) numbered milestones
col = {"early": "#9E9E9E", "meg": "#4C72B0", "late": "#DD8452"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 2.25), gridspec_kw={"width_ratios": [1.15, 1]})
for g, lab_, mk in (("meg", "before the in-domain LM", "o"), ("late", "after the in-domain LM", "s")):
    ks = [k for k in pre if st[k] == g]
    a1.scatter([S[k][0] for k in ks], [S[k][1] for k in ks], s=12, c=col[g], marker=mk, label=lab_, alpha=0.85, edgecolors="none")
a1.scatter([S[k][0] for k in ACTUAL], [S[k][1] for k in ACTUAL], s=42, facecolors="none", edgecolors="k", linewidths=0.8, label="final selection")
for k, t, off in (("final_V", "V", (4, 2)), ("final_gpt2", "best private", (-30, 6))):
    a1.annotate(t, (S[k][0], S[k][1]), xytext=off, textcoords="offset points", fontsize=7)
lim = (0.52, 0.745); a1.plot(lim, lim, ls=":", c="0.6", lw=0.8)
a1.set_xlim(lim); a1.set_ylim(0.52, 0.705); a1.set_xlabel("public BAcc@10"); a1.set_ylabel("private BAcc@10")
a1.legend(fontsize=7, loc="lower right", frameon=False, handletextpad=0.2); a1.set_title("(a) 91 pre-deadline submissions", fontsize=7.5)
M = [("submission_deep_final", "linear baseline"), ("sub_v1", "per-word CNN"), ("sub_c1", "sentence Transformer"),
     ("sub_c5", "+ repeated-sentence handling"), ("sub_c5_lmpmi05", "+ closed-vocab LM fusion"),
     ("final_gpt2", "9-model ensemble (best private)"), ("finalW_calib2", "3 s windows, LM calibration"),
     ("final_P1", "+ self-training"), ("final_B5", "diverse ensemble B5"), ("final_B5i_L1", "+ in-domain LM (807, seed 0)"),
     ("final_IN1_Q_big", "IN1 + LM on 3,742 sentences"), ("final_IN1Qp_SCGL_b0.5", "+ candidate decoding")]
xs = np.arange(1, len(M) + 1)
a2.plot(xs, [S[k][0] for k, _ in M], "o-", c="#4C72B0", ms=3, lw=1.1, label="public")
a2.plot(xs, [S[k][1] for k, _ in M], "s-", c="#C44E52", ms=3, lw=1.1, label="private")
a2.set_xticks(xs); a2.set_xlabel("milestone (see caption)"); a2.set_ylabel("BAcc@10")
a2.legend(fontsize=7, frameon=False, loc="lower right"); a2.set_title("(b) development milestones", fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT / "fig_overview.pdf"); fig.savefig(H / "data/fig_overview.png", dpi=130)
out_m = [t for _, t in M]

# Figure 3: (a) paired effects, (b) selection back-test
fig, (b1, b2) = plt.subplots(1, 2, figsize=(5.5, 2.35), gridspec_kw={"width_ratios": [1.15, 1]})
b1.axvspan(-0.03, 0.03, color="0.93", zorder=0); b1.axhspan(-floor, floor, color="0.86", zorder=0, lw=0)
b1.axhline(0, c="0.6", lw=0.6); b1.axvline(0, c="0.6", lw=0.6)
KEPT, FLIP = "#1F3A68", "#E8686B"                              # dark and light, so the two stay distinct in grayscale
for p in pairs:
    if abs(p["d_pub"]) > 0.2:
        continue
    c = FLIP if np.sign(p["d_pub"]) != np.sign(p["d_priv"]) else KEPT
    b1.scatter(p["d_pub"], p["d_priv"], marker="o" if p["kind"] == "single" else "^", s=16, c=c, edgecolors="none")
NOTE = {"closed-vocab LM fusion (PMI)": ("LM fusion", (-40, -3)), "in-domain LM (807 sentences, seed 0)": ("in-domain LM", (-58, -3)),
        "in-domain corpus 807 -> 1,368": ("larger corpus", (4, -2)), "sentence-end LM term": ("sentence end", (-28, 5)),
        "validation-motivated stack V": ("V", (3, 3)), "Qwen3-8B instead of GPT-2": ("Qwen3-8B", (5, 2))}
for p in pairs:
    if p["change"] in NOTE:
        t, off = NOTE[p["change"]]
        b1.annotate(t, (p["d_pub"], p["d_priv"]), xytext=off, textcoords="offset points", fontsize=6.5)
b1.legend(handles=[Line2D([], [], ls="", marker="o", c=KEPT, ms=4, label="sign kept"),
                   Line2D([], [], ls="", marker="o", c=FLIP, ms=4, label="sign flipped"),
                   Line2D([], [], ls="", marker="^", c="0.5", ms=4, label="bundle")],
          fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, handletextpad=0.1, columnspacing=0.8)
b1.set_xlabel(r"$\Delta$ public"); b1.set_ylabel(r"$\Delta$ private"); b1.set_title("(a) reference-variant comparisons", fontsize=7.5)
b2.hist(draws, bins=np.arange(0.655, 0.6905, 0.0025), color="#4C72B0", alpha=0.6, label="10 random submissions")
for v, t, c, ls in ((x0, "actual selection", "#C44E52", "--"), (out["select_top10_public_not_late"][0], "top-10 public before in-domain LM", "0.3", ":"),
                    (out["select_earlier"][0], "day-before selection", "#2E8B57", "-.")):
    b2.axvline(v, c=c, lw=1.2, ls=ls, label=t)
b2.legend(fontsize=7, frameon=False, loc="upper left"); b2.set_ylim(0, 9000)
b2.set_xlabel("best private of ten"); b2.set_ylabel("random draws"); b2.set_title("(b) final-selection back-test", fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT / "fig_pairs.pdf"); fig.savefig(H / "data/fig_pairs.png", dpi=130)
json.dump(out_m, open(H / "data/milestones.json", "w", encoding="utf-8"), ensure_ascii=False)
print("figures written")
