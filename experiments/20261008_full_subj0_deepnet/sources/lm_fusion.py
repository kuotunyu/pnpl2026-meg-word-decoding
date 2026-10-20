"""Language-model fusion for the holdout sentence partition.

Each holdout sentence is a sequence of words from the closed 50-word vocabulary. Combine the MEG
model's per-word log-probabilities with a pretrained causal LM (GPT-2 family) restricted to that
vocabulary: beam search over the sentence, then per-position marginals from the N-best list.
Isolated-word rows keep the MEG distribution unchanged.

  python lm_fusion.py --tags c5_dedup --alpha 0.5 --out ../outputs/sub_x.csv [--lm gpt2] [--beam 128]
"""
import argparse, csv, os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
COMP = HERE.parents[2]
VOCAB = ["is", "the", "a", "to", "it", "i", "not", "was", "we", "be", "he", "that", "have", "this", "they", "of",
         "there", "and", "are", "in", "but", "will", "so", "all", "my", "for", "she", "were", "any", "really", "at",
         "out", "our", "am", "it's", "had", "him", "an", "very", "has", "do", "can", "time", "think", "good",
         "always", "new", "people", "as", "on"]


def hl_path(t):
    """holdout logits of a tag; PNPL_RS=1 -> the input-rescaled ones (infer_rescale.py) when they exist"""
    d = HERE.parent / "outputs" / t
    return d / "holdout_logits_rs.npz" if os.environ.get("PNPL_RS") and (d / "holdout_logits_rs.npz").exists() else d / "holdout_logits.npz"


def form(w, first):
    if w == "i":
        return "I" if first else " I"
    return w.capitalize() if first else " " + w


class LM:
    def __init__(self, name, device):
        self.tok = AutoTokenizer.from_pretrained(name)
        dt = torch.float16 if device == "cuda" else torch.float32
        self.m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dt).to(device).eval()
        self.dev = device
        self.bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        self.ids = {first: [self.tok.encode(form(w, first), add_special_tokens=False) for w in VOCAB] for first in (True, False)}

    @torch.no_grad()
    def next_logp(self, prefixes):
        """prefixes: list of token-id lists (without BOS). Returns (len, 50) log P(word | prefix)."""
        first = [len(p) == 0 for p in prefixes]
        out = np.zeros((len(prefixes), 50))
        base = self._last_logp([[self.bos] + p for p in prefixes])
        for r, f in enumerate(first):
            ids = self.ids[f]
            out[r] = [base[r, t[0]] for t in ids]
        # multi-token words: add the remaining token log-probs
        for wi in range(50):
            for f in (True, False):
                t = self.ids[f][wi]
                if len(t) == 1:
                    continue
                rows = [r for r in range(len(prefixes)) if first[r] == f]
                if not rows:
                    continue
                seqs = [[self.bos] + prefixes[r] + t[:k] for r in rows for k in range(1, len(t))]
                lp = self._last_logp(seqs)
                j = 0
                for r in rows:
                    for k in range(1, len(t)):
                        out[r, wi] += lp[j, t[k]]; j += 1
        return out

    @torch.no_grad()
    def seq_loglik(self, wordseqs, bs=512):
        """Total log-prob of each closed-vocabulary word sequence (list of word ids)."""
        seqs = [[self.bos] + sum((self.ids[k == 0][w] for k, w in enumerate(ws)), []) for ws in wordseqs]
        out = np.zeros(len(seqs))
        order = np.argsort([len(q) for q in seqs])
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]; L = max(len(seqs[j]) for j in idx)
            ids = torch.full((len(idx), L), self.tok.eos_token_id); att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, j in enumerate(idx):
                ids[r, :len(seqs[j])] = torch.tensor(seqs[j]); att[r, :len(seqs[j])] = 1
            ids, att = ids.to(self.dev), att.to(self.dev)
            lp = torch.log_softmax(self.m(input_ids=ids, attention_mask=att).logits.float(), -1)
            tl = (lp[:, :-1].gather(2, ids[:, 1:, None])[..., 0] * att[:, 1:]).sum(1).cpu().numpy()
            out[idx] = tl
        return out

    def _last_logp(self, seqs):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.tok.eos_token_id)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
        res = []
        for i in range(0, len(seqs), 256):
            lo = self.m(input_ids=ids[i:i + 256].to(self.dev), attention_mask=att[i:i + 256].to(self.dev)).logits.float()
            last = att[i:i + 256].sum(1) - 1
            res.append(torch.log_softmax(lo[torch.arange(len(last)), last.to(self.dev)], -1).cpu())
        return torch.cat(res).numpy()


@torch.no_grad()
def eos_logp(lm, prefixes):
    """(len, 50): log P(sentence end '.' | prefix + word) for each closed-vocabulary word."""
    if not hasattr(lm, "eos_id"):
        lm.eos_id = lm.tok.encode(".", add_special_tokens=False)[0]
    out = np.zeros((len(prefixes), 50))
    seqs = [[lm.bos] + p + lm.ids[len(p) == 0][w] for p in prefixes for w in range(50)]
    lp = lm._last_logp(seqs)[:, lm.eos_id].reshape(len(prefixes), 50)
    return lp


def fuse_sentence(lm, meg, alpha, beam, soft=False, bidir=0, eos=False):
    """meg: (L, 50) log-probs. Returns (L, 50) marginal probabilities from the final N-best, or with
    soft=True: softmax(meg_j + alpha * log sum_h pi_h P_LM(.|prefix_h)) using the beam over prefixes."""
    hyps = [([], [], 0.0)]  # (word ids, token ids, score)
    softout = np.zeros_like(meg)
    for j in range(len(meg)):
        lp = lm.next_logp([h[1] for h in hyps])
        if eos and j == len(meg) - 1:  # last word: the sentence ends after it -> P(w | prefix, end) (renormalised)
            lp = lp + eos_logp(lm, [h[1] for h in hyps])
            lp = lp - np.log(np.exp(lp - lp.max(1, keepdims=True)).sum(1, keepdims=True)) - lp.max(1, keepdims=True)
        sc = np.array([h[2] for h in hyps]); pi = np.exp(sc - sc.max()); pi /= pi.sum()
        softout[j] = np.log((pi[:, None] * np.exp(lp)).sum(0))  # LM predictive log-dist at j
        cand = []
        for (ws, ts, sc), row in zip(hyps, lp):
            tot = sc + alpha * row + meg[j]
            for wi in np.argsort(-tot)[:beam]:
                cand.append((tot[wi], ws, ts, wi))
        cand.sort(key=lambda c: -c[0])
        hyps = []
        for tot, ws, ts, wi in cand[:beam]:
            hyps.append((ws + [wi], ts + lm.ids[len(ws) == 0][wi], tot))
    if bidir:
        # whole-sentence LM evidence with the other positions taken from the top-K joint hypotheses
        top = hyps[:bidir]
        sc = np.array([h[2] for h in top]); pi = np.exp(sc - sc.max()); pi /= pi.sum()
        L = len(meg); seqs = []
        for ws, _, _ in top:
            for j in range(L):
                for w in range(50):
                    v = list(ws); v[j] = w; seqs.append(v)
        ll = lm.seq_loglik(seqs).reshape(len(top), L, 50)
        m = ll.max(0, keepdims=True)
        mix = m[0] + np.log((pi[:, None, None] * np.exp(ll - m)).sum(0))
        return mix - np.log(np.exp(mix - mix.max(1, keepdims=True)).sum(1, keepdims=True)) - mix.max(1, keepdims=True)
    if soft:
        return softout
    sc = np.array([h[2] for h in hyps]); p = np.exp(sc - sc.max()); p /= p.sum()
    marg = np.zeros_like(meg)
    for (ws, _, _), pi in zip(hyps, p):
        marg[np.arange(len(ws)), ws] += pi
    return marg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beam", type=int, default=128)
    ap.add_argument("--lm", default="gpt2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--soft", action="store_true")
    ap.add_argument("--pmi", action="store_true")
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--bidir", type=int, default=0)
    ap.add_argument("--eos", action="store_true", help="use that the sentence ends after its last word")
    a = ap.parse_args()
    H = [np.load(hl_path(t)) for t in a.tags]
    lp = np.mean([h["logp"] - a.tau * np.log(h["prior"]) for h in H], 0)
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    if os.environ.get("PNPL_DUR"):  # word-timing evidence for the sentence rows (duration_prior.py holdout)
        E = np.load(os.environ["PNPL_DUR"]); lp[:len(E)] += float(os.environ.get("PNPL_DUR_BETA", "0.5")) * E
        lp = lp - lp.max(1, keepdims=True); lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    wm = np.load(COMP / "data/raw/LibriBrain-Competition-2026/COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz")["word_mask"]
    lm = LM(a.lm, a.device)
    probs = np.exp(lp)
    off = 0; mixes = []
    for si, r in enumerate(wm):
        n = int(r.sum())
        marg = fuse_sentence(lm, lp[off:off + n], a.alpha, a.beam, a.soft, a.bidir, a.eos)
        if a.soft:
            mixes.append(marg)
        else:
            # words outside the N-best keep their MEG ordering below every N-best word
            probs[off:off + n] = marg + 1e-4 * np.exp(lp[off:off + n])
        off += n
        if si % 40 == 0:
            print(si, flush=True)
    if a.soft:
        mix = np.concatenate(mixes)
        np.save(str(a.out).replace(".csv", "_lmmix.npy"), mix)
        bg = np.log(np.exp(mix).mean(0)) if a.pmi else 0.0   # LM marginal over all holdout positions
        z = lp[:off] + a.alpha * (mix - bg)
        probs[:off] = np.exp(z - z.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    header = open(COMP / "data/raw/deep_example_submission.csv", encoding="utf-8").readline().strip().split(",")[:51]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header)
        for i, row in enumerate(probs):
            w.writerow([i] + [f"{v:.6f}" for v in row])
    np.save(str(a.out).replace(".csv", "_probs.npy"), probs)
    print("wrote", a.out, probs.shape)


if __name__ == "__main__":
    main()
