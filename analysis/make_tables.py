"""Write the paired-comparison tables of the paper from data/analysis.json (no hand-copied numbers).
  python make_tables.py  ->  ../paper/tab_pairs_main.tex, ../paper/tab_pairs_all.tex
"""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
A = json.load(open(H / "data/analysis.json", encoding="utf-8"))
P = {p["change"]: p for p in A["pairs"]}
import re
tex = lambda s: re.sub(r"(tau|beta|lambda)([0-9.]+)", lambda m: "$\\" + m.group(1) + "{=}" + m.group(2) + "$",
                       s.replace("_", r"\_").replace("->", r"$\to$"))
fmt = lambda x: f"${x:+.3f}$"
VAL = {"+": "$+$", "-": "$-$", "n/a": "n/a"}

MAIN = ["sentence-context model (redesign)", "closed-vocab LM fusion (PMI)", "Qwen3-8B instead of GPT-2",
        "N-best hard marginals instead of soft PMI", None,
        "self-training round 1", "LM fusion after self-training", "all members self-trained (one teacher)",
        "sentence-end LM term", "sentence prior correction tau0.3", "word-timing evidence beta0.5", "validation-motivated stack V", None,
        "in-domain LM (807 sentences)", "in-domain corpus 807 -> 1,368", "in-domain corpus 807 -> 2,103 (IN1)",
        "generic candidates, lambda0.15", "candidate decoding (guided pool + LM prior)"]
rows = []
for c in MAIN:
    if c is None:
        rows.append(r"    \midrule")
        continue
    p = P[c]
    k = "" if p["kind"] == "single" else r"$^\dagger$"
    rows.append(f"    {tex(c)}{k} & {fmt(p['d_pub_3dp'])} & {fmt(p['d_priv_3dp'])} & {VAL[p['val']]} \\\\")
head = [r"\begin{tabular}{lrrc}", r"    \toprule", r"    Change & $\Delta$ public & $\Delta$ private & Validation \\", r"    \midrule"]
tail = [r"    \bottomrule", r"\end{tabular}"]
(H.parent / "paper/tab_pairs_main.tex").write_text("\n".join(head + rows + tail) + "\n", encoding="utf-8")

rows = []
for p in A["pairs"]:
    k = "S" if p["kind"] == "single" else "B"
    rows.append(f"    {tex(p['change'])} & \\texttt{{{tex(p['base'])}}} & \\texttt{{{tex(p['variant'])}}} & {k} & "
                f"{fmt(p['d_pub_3dp'])} & {fmt(p['d_priv_3dp'])} & {VAL[p['val']]} \\\\")
head = [r"\begin{tabular}{lllcrrc}", r"\toprule", r"Change & Reference & Variant & & $\Delta$ pub. & $\Delta$ priv. & Val. \\", r"\midrule"]
(H.parent / "paper/tab_pairs_all.tex").write_text("\n".join(head + rows + [r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")
print("tables written:", len(MAIN) - MAIN.count(None), "main rows,", len(A["pairs"]), "appendix rows")
