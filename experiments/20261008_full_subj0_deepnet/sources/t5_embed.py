"""Static T5-large encoder embeddings (middle layer, mean over word tokens) for every word seen >= 3 times.
Output: data/derived/t5_word_emb.npz with words (N,) and emb (N, 1024), L2-normalised."""
import glob
from collections import Counter
from pathlib import Path
import numpy as np, torch
from transformers import T5EncoderModel, T5TokenizerFast, T5Tokenizer

COMP = Path(__file__).resolve().parents[3]
T5 = COMP / "data/raw/google-t5_t5-large_150ebc2c4b72291e770f58e6057481c8d2ed331a"
cnt = Counter()
for f in glob.glob(str(COMP / "data/derived/subj0_pca_epochs/runs/*.npz")):
    cnt.update(np.load(f, allow_pickle=True)["word"].tolist())
words = [w for w, c in cnt.items() if c >= 3 and w]
print(len(words), "words", flush=True)
tok = T5Tokenizer.from_pretrained(str(T5))
m = T5EncoderModel.from_pretrained(str(T5), torch_dtype=torch.float32).eval()
layer = 12
E = []
with torch.no_grad():
    for i in range(0, len(words), 256):
        b = words[i:i + 256]
        enc = tok(b, return_tensors="pt", padding=True)
        hs = m(**enc, output_hidden_states=True).hidden_states[layer]
        mask = enc["attention_mask"].clone()
        mask[torch.arange(len(b)), mask.sum(1) - 1] = 0  # drop </s>
        v = (hs * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        E.append(torch.nn.functional.normalize(v, dim=1).numpy())
        if i % 2560 == 0:
            print(i, flush=True)
np.savez(COMP / "data/derived/t5_word_emb.npz", words=np.array(words), emb=np.concatenate(E).astype(np.float32))
print("saved")
