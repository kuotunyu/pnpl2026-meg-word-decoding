# MEG word decoding with sentence-level language priors (PNPL 2026, Deep Track)

This repository contains the code for team `yu_kuo14` in the [PNPL Competition 2026: Deep Track](https://www.kaggle.com/competitions/pnpl-competition-2026-deep). The task is to decode which of 50 common English words a single participant hears from MEG recordings. Submissions are scored by class-balanced top-10 accuracy (BAcc@10, chance = 0.20).

**Result:** our best public-leaderboard score is 0.733 (13th of 90 teams on 2026-10-20). The private leaderboard and the organisers' verification are pending.

A workshop write-up is in preparation (due 1 December 2026) and will be added under `paper/`.

## Pipeline

1. **MEG decoder.** A per-word CNN encoder followed by a sentence-level Transformer, after d'Ascoli et al. (2025). The ensemble has about 30 members, varying in window, sampling rate, depth, width and normalisation. Some members are self-trained on the unlabelled test recordings with soft targets.
2. **Closed-vocabulary LM fusion.** A causal LM scores the 50 words at each sentence position, using soft beam-prefix mixtures. Its PMI (LM log-probability minus its average over test positions) is added to the MEG log-probabilities with α = 0.5. Public score: 0.584 → 0.630.
3. **In-domain LM.** GPT-2 is fine-tuned on synthetic sentences that use only the 50 words, written by Claude and by Qwen3.5-9B (see `corpora/`). On a fixed ensemble this gave +0.010 (0.710 → 0.720).
4. **Whole-sentence candidate decoding.** Every candidate sentence of the right length is scored by its summed MEG log-probability plus 0.5 × the in-domain GPT-2 sentence log-probability. The resulting per-position marginals are mixed into the submission with λ = 0.3. Candidates include LLM proposals guided by our own top-6 words per position. This gave 0.724 → 0.733, but **post-deadline ablations show that the gain is not reliable** (see the table below).

| Candidate-decoding ablation (base 0.724, λ = 0.3) | Public |
|---|---|
| general pool, no LM prior | 0.721 |
| general pool, LM prior | 0.706 |
| guided pool, no LM prior | 0.708 |
| guided pool, LM prior (best submission) | 0.733 |
| guided pool, LM prior only (no MEG term) | 0.708 |

## Repository layout

```
experiments/20261008_full_subj0_deepnet/sources/   code (run from experiments/20261008_full_subj0_deepnet/)
  download_subj0.py, preprocess.py, preprocess_w3.py  data download and PCA epochs (1 s / 3 s windows)
  train.py, train_ctx.py                               MEG models (CNN + sentence Transformer)
  pseudo_ft.py, teacher_W.csv, teacher_P1.csv          self-training on the test recordings
  lm_fusion.py, lm_calib.py, make_final.py             LM fusion and submission assembly
  vocab_corpus.py, gen_vocab_llm.py, ft_vocab_lm.py    in-domain corpus and LM
  gen_guided.py, sent_cand.py                          whole-sentence candidate decoding
corpora/        vocabulary-only sentence corpora used for the LMs and candidates
docs/REPRODUCE.md  ensemble members, training commands, checkpoints and the commands for the best submissions
```

The scripts expect the competition data under `data/raw/LibriBrain-Competition-2026/` at the repository root, and they write to `experiments/20261008_full_subj0_deepnet/outputs/`. Neither folder is tracked. `download_subj0.py` fetches the data from Hugging Face.

## Notes

- No labels of the test set, cross-track metadata or timing leaks are used. Self-training and guided proposals use only our own predictions.
- Model checkpoints and fine-tuned LMs are on [Google Drive](https://drive.google.com/drive/folders/1HIGSl6Yx3K_mzwb7Op_ZBc0fdD9w8FNT); see `docs/REPRODUCE.md`, Section 5.

## License

MIT (code). The corpora in `corpora/` were generated for this project and are released under the same license. The competition data is not included; see the competition's own terms.
