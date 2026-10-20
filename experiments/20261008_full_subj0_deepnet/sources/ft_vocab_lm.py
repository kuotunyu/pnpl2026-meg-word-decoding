"""Fine-tune GPT-2 on the synthetic in-domain sentences (vocab_corpus.py) so the closed-vocabulary LM prior
matches short conversational utterances built from the 50 competition words. Text is formatted exactly like
lm_fusion.form(): first word capitalised, "I" capitalised, sentences wrapped in <|endoftext|>.
  python ft_vocab_lm.py --base gpt2 --epochs 3 --lr 5e-5 --out ../outputs/lm_gpt2_vocab
"""
import argparse, random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_fusion import form

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="gpt2")
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--out", required=True)
ap.add_argument("--corpus", default="vocab_corpus.txt")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)
sents = Path(__file__).resolve().parents[1].joinpath("outputs", a.corpus).read_text(encoding="utf-8").split("\n")
sents = [s for s in sents if s.strip()]
tok = AutoTokenizer.from_pretrained(a.base); eos = tok.eos_token_id
texts = ["".join(form(w, i == 0) for i, w in enumerate(s.split())) for s in sents]
data = [[eos] + tok.encode(t, add_special_tokens=False) + [eos] for t in texts]
m = AutoModelForCausalLM.from_pretrained(a.base).cuda().train()
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
steps = int(len(data) / a.bs * a.epochs); print(len(data), "sentences", steps, "steps; example:", texts[0], "|", texts[100])
for st in range(steps):
    b = random.sample(data, a.bs); L = max(map(len, b))
    x = torch.full((a.bs, L), eos); att = torch.zeros((a.bs, L), dtype=torch.long)
    for i, s in enumerate(b):
        x[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
    lab = x.clone(); lab[att == 0] = -100
    loss = m(input_ids=x.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss
    opt.zero_grad(); loss.backward(); opt.step()
    if st % 50 == 0 or st == steps - 1:
        print(st, round(loss.item(), 3), flush=True)
m.save_pretrained(a.out); tok.save_pretrained(a.out); print("saved", a.out)
