"""Merge vocab_corpus.txt (batch 1) and sources/vocab_corpus2.txt (batch 2, also written by Claude) into
outputs/vocab_corpus_big.txt, keeping only sentences made of the 50 competition words."""
from collections import Counter
from pathlib import Path
from vocab_corpus import VOCAB

H = Path(__file__).resolve().parent
old = [l.strip() for l in (H.parent / "outputs" / "vocab_corpus.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
new = [l.strip().lower() for l in (H / "vocab_corpus2.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
ok = [l for l in new if all(w in VOCAB for w in l.split())]
bad = [l for l in new if l not in ok]
allc = list(dict.fromkeys(old + ok))
(H.parent / "outputs" / "vocab_corpus_big.txt").write_text("\n".join(allc) + "\n", encoding="utf-8")
print("batch2 kept", len(ok), "dropped", len(bad), bad[:15])
print("total unique", len(allc), "length dist", sorted(Counter(len(s.split()) for s in allc).items()))
