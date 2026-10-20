"""MEG-guided sentence proposals: for each holdout sentence, an instruct LLM is shown the top-k words per
position from our own submission and asked for grammatical sentences of exactly that length over the 50
words. Outputs are filtered to vocabulary-only sentences of the right length and pooled as extra candidates
for sent_cand.py. Uses only our predictions (no labels).
  python gen_guided.py --sub ../outputs/final_IN1_Q_llm_plus.csv --k 6 --n_per 3 --out ../outputs/vocab_corpus_guided.txt
"""
import argparse, random, re
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_fusion import VOCAB

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--sub", required=True)
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--n_per", type=int, default=3, help="prompts per holdout sentence")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)
V = set(VOCAB)
P = pd.read_csv(a.sub).values[:, 1:].astype(float)
try:
    wm = np.load(HERE.parents[2] / "data/raw/LibriBrain-Competition-2026/COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz")["word_mask"]
except Exception:
    wm = np.loadtxt(HERE / "holdout_wordmask.csv", delimiter=",").astype(bool)
lens = wm.sum(1).astype(int)
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()


def prompt(seg):
    n = len(seg)
    rows = []
    for j in range(n):
        top = [VOCAB[i] for i in np.argsort(-seg[j])[:a.k]]
        random.shuffle(top)
        rows.append(f"word {j + 1}: {' / '.join(top)}")
    msg = (f"Write 15 different natural, grammatical English sentences of EXACTLY {n} words each, one per line, "
           f"lowercase, no punctuation. Use ONLY these words: {', '.join(sorted(V))}. "
           f"For each position, prefer (but you may change some) these likely words:\n" + "\n".join(rows) +
           "\nOutput only the sentences.")
    try:
        return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)


jobs, off = [], 0
for n in lens:
    if n >= 3:
        jobs += [(n, P[off:off + n])] * a.n_per
    off += n
keep, seen, total = [], set(), 0
for i in range(0, len(jobs), a.bs):
    batch = jobs[i:i + a.bs]
    enc = tok([prompt(s) for _, s in batch], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=350, do_sample=True, temperature=0.9, top_p=0.95)
    for (n, _), o in zip(batch, out[:, enc["input_ids"].shape[1]:]):
        for line in tok.decode(o, skip_special_tokens=True).splitlines():
            s = re.sub(r"^[\s\d\.\-\*\)]+", "", line).strip().lower()
            s = re.sub(r"[^a-z' ]", "", s.replace("’", "'")).strip()
            if not s:
                continue
            total += 1
            ws = s.split()
            if len(ws) == n and all(w in V for w in ws) and s not in seen:
                seen.add(s); keep.append(s)
    print(f"{i + len(batch)}/{len(jobs)} prompts: {len(keep)} kept of {total} lines", flush=True)
Path(a.out).write_text("\n".join(keep) + "\n", encoding="utf-8")
print("wrote", a.out, len(keep))
