"""Generate in-domain sentences with an instruction-tuned LLM (the holdout utterances are likely LLM-written
short sentences over the 50 competition words; arXiv 2609.03231 App. B examples). Only the public vocabulary
and the paper's example sentences are given to the model; outputs are filtered to vocabulary-only sentences.
  python gen_vocab_llm.py --model Qwen/Qwen3.5-9B --n_prompts 300 --out ../outputs/vocab_corpus_llm.txt
"""
import argparse, random, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vocab_corpus import VOCAB

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--n_prompts", type=int, default=300)
ap.add_argument("--bs", type=int, default=12)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--lens", default="3 to 8")
a = ap.parse_args()
ap2 = None
random.seed(a.seed); torch.manual_seed(a.seed)
words = sorted(VOCAB)
EX = ["i think so", "we have time", "i am very good", "i am not good at all", "can i have this", "we can do this"]
STYLES = ["short everyday conversational sentences", "simple statements someone might say to a friend",
          "short sentences about people and time", "simple sentences a person might say about how they feel or think",
          "short questions and answers", "simple sentences about what people have or will do",
          "very simple sentences like in a beginner English book", "short sentences someone could say in a hospital or at home"]
tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()


LENS = a.lens

def prompt():
    ws = words[:]; random.shuffle(ws)
    ex = random.sample(EX, 3)
    msg = (f"Write 25 different {random.choice(STYLES)} in English, one per line, lowercase, no punctuation. "
           f"Use ONLY these words and no other words at all: {', '.join(ws)}. "
           f"Each sentence has {LENS} words. Examples: {' / '.join(ex)}. Output only the sentences.")
    try:
        return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)


keep, seen, total = [], set(), 0
for i in range(0, a.n_prompts, a.bs):
    ps = [prompt() for _ in range(a.bs)]
    enc = tok(ps, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=400, do_sample=True, temperature=a.temp, top_p=0.95)
    for o in out[:, enc["input_ids"].shape[1]:]:
        for line in tok.decode(o, skip_special_tokens=True).splitlines():
            s = re.sub(r"^[\s\d\.\-\*\)]+", "", line).strip().lower()
            s = re.sub(r"[^a-z' ]", "", s.replace("’", "'")).strip()
            if not s:
                continue
            total += 1
            ws = s.split()
            if 2 <= len(ws) <= 10 and all(w in VOCAB for w in ws) and s not in seen:
                seen.add(s); keep.append(s)
    print(f"{i + a.bs}/{a.n_prompts} prompts: {len(keep)} kept of {total} lines", flush=True)
Path(a.out).write_text("\n".join(keep) + "\n", encoding="utf-8")
print("wrote", a.out, len(keep))
