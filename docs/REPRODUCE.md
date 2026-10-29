# Reproducing the submissions and the write-up

Unless stated otherwise, run the commands with bash from `experiments/20261008_full_subj0_deepnet/`. Set `PY` to your Python interpreter (default `python`). The scripts read the competition data from `data/raw/` at the repository root and write to `experiments/20261008_full_subj0_deepnet/outputs/`. Neither folder is tracked.

## 0. Environment

`requirements.txt` lists the package versions of the local environment (Python 3.12.13) used for the RTX 4090 runs and the analysis. The Colab A100 runs used Colab's preinstalled packages in October 2026. The analysis and the paper need only NumPy and matplotlib, plus [Tectonic](https://tectonic-typesetting.github.io/) (we used 0.17.0) to build the PDF.

## 1. Data

```bash
python sources/download_subj0.py        # subj0 runs and events from pnpl/LibriBrain and pnpl/LibriBrain2 -> data/raw/
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="../../data/raw/LibriBrain-Competition-2026")
PY
kaggle competitions download pnpl-competition-2026-deep -f deep_example_submission.csv -p ../../data/raw
```

The last file comes from the competition's Data page (it can also be downloaded in the browser). The scripts read only its header, which gives the submission column names.

Preprocessing:

```bash
python sources/preprocess.py pca                    # global PCA 306 -> 80, fitted on a sample of runs
python sources/preprocess.py runs 8                 # 1 s epochs at 125 Hz -> data/derived/subj0_pca_epochs (8 worker processes)
python sources/preprocess.py holdout
python sources/preprocess_w3.py runs 8              # 3 s epochs [-0.5, 2.5] s at 50 Hz -> data/derived/subj0_pca_w3
python sources/preprocess_w3.py holdout
PNPL_FS=100 python sources/preprocess_w3.py runs 8  # 100 Hz variant -> data/derived/subj0_pca_w3h
PNPL_FS=100 python sources/preprocess_w3.py holdout
```

The seven validation runs (`sources/train.py`) are held out from supervised training, but the PCA sample includes two of them.

## 2. MEG models

Every ensemble member is a tag. Every run writes `outputs/<tag>/model.pt` and `outputs/<tag>/holdout_logits.npz`.

- **Originally trained members** are produced by `python sources/train_ctx.py --tag <tag> --full ...`. 3 s members use `--data subj0_pca_w3 --w3aug` (or `subj0_pca_w3h` with `PNPL_FS=100`). The exact arguments of every member are stored in its checkpoint (Section 5).
- **Self-trained members** are produced by `sources/pseudo_ft.py`. Its defaults are lr 1e-4, 3 passes over the training data and 32 test segments per step:

| members | command |
|---|---|
| pw3_s1-3, pf7_s1 | `python sources/pseudo_ft.py --tag pw3_sN --init w3f_sN --teacher sources/teacher_W.csv --data subj0_pca_w3 --w3aug --seed N` (pf7_s1: `--init f7_s1 --data subj0_pca_epochs --seed 4`) |
| rw3_s7, rw3_s8, rw3e_s12, rw3d_s13 | `PNPL_FS=50 python sources/pseudo_ft.py --tag <tag> --init <w3f_s7/w3f_s8/w3ef_s12/w3df_s13> --teacher sources/teacher_P1.csv --data subj0_pca_w3 --w3aug --seed <7/8/12/13>` |
| rw3h_s1-3 | `PNPL_FS=100 python sources/pseudo_ft.py --tag rw3h_sN --init w3hf_sN --teacher sources/teacher_P1.csv --data subj0_pca_w3h --w3aug --seed 2N` (seeds 21-23) |
| rf7e_s4, rf7e_s5 | `PNPL_FS=125 python sources/pseudo_ft.py --tag <tag> --init <f7e_s4/f7e_s5> --teacher sources/teacher_P1.csv --data subj0_pca_epochs --seed <14/15>` |
| pfl_s1-3 | `python sources/pseudo_ft.py --tag pfl_sN --init f_sN --teacher sources/teacher_P1.csv --data subj0_pca_epochs --seed 3N` (seeds 31-33) |

The teacher files are earlier submissions of ours (public 0.679 and 0.694).

Naming conventions:
- `w3*`: 3 s models; `f*`: 1 s models.
- `p*`, `q*`, `r*`, `s*`, `t*`, `u*`: self-trained members.
- `L` and `M`: 12-layer models; `X`: the 16-layer model; `C`: the wide CNN; `I` and `G`: instance- and window-normalised models.

### Ensemble member lists

**IN1, sentence rows** (31 models; the base of the best submissions):
```
pw3_s1 pw3_s2 pw3_s3 pf7_s1 f7_s2 f7_s3 rw3_s7 rw3_s8 rw3e_s12 rw3d_s13 rw3h_s1 rw3h_s2 rw3h_s3 rf7e_s4 rf7e_s5
pfl_s1 pfl_s2 pfl_s3 w3e_s14 w3e_s15 w3hf_s4 f7e_s6 fe_s4 w3L_s21 w3C_s22 w3f_s4 w3f_s5 w3f_s6 w3I_s61 w3G_s62 w3I_s63
```

**Isolated-word rows** (used by every final submission; repeated tags are up-weighted). These rows appear not to be scored (see the write-up), so this list does not affect any score:
```
w3f_s1 w3f_s2 w3f_s3 w3f_s4 w3f_s5 w3f_s6 w3f_s7 w3f_s8 w3hf_s1 w3hf_s2 w3hf_s3 w3ef_s12 w3df_s13 w3e_s14 w3e_s15 w3hf_s4 w3L_s21 w3C_s22
f7_s1 f7_s2 f7_s3 f_s1 f_s2 f_s3 f7e_s4 f7e_s5 f7e_s6 fe_s4   (this 1 s group listed twice)
```

## 3. In-domain LMs

```bash
cp ../../corpora/*.txt outputs/
$PY sources/ft_vocab_lm.py --base gpt2 --corpus vocab_corpus_llm_plus.txt --out outputs/lm_gpt2_llm_plus   # 2,103 sentences
$PY sources/ft_vocab_lm.py --base gpt2 --corpus vocab_corpus_llm_big.txt  --out outputs/lm_gpt2_llm_big    # 3,742 sentences
```

Defaults: 3 epochs, lr 5e-5, batch 16, seed 0.

Corpora (`corpora/`, one sentence per line, only the 50 target words):
- `vocab_corpus.txt`: 807 sentences written by Claude (`sources/vocab_corpus.py`).
- `vocab_corpus_big.txt`: 1,368 sentences, the 807 plus a second batch written by Claude.
- `vocab_corpus_llm.txt`, `vocab_corpus_llm2.txt`, `vocab_corpus_llm_long.txt`: generated by Qwen3.5-9B (`sources/gen_vocab_llm.py`).
- `vocab_corpus_llm_plus.txt` (2,103) is the union of `vocab_corpus.txt` and `vocab_corpus_llm.txt`; `vocab_corpus_llm_big.txt` (3,742) also adds `vocab_corpus_llm2.txt`.
- `vocab_corpus_guided.txt`: the MEG-guided proposals used by the best public submission (`sources/gen_guided.py`).

## 4. Best submissions

`S` is the IN1 list above and `ISO` is the isolated-word list. These commands use the released corpora, including the guided proposals.

```bash
# public 0.724: IN1 + in-domain GPT-2 (2,103) + GPT-2-medium
$PY sources/lm_fusion.py --tags $S --lm outputs/lm_gpt2_llm_plus --alpha 0.5 --beam 32 --soft --pmi --device cuda --out outputs/fused_IN1_llm_plus.csv
$PY sources/lm_fusion.py --tags $S --lm gpt2-medium            --alpha 0.5 --beam 32 --soft --pmi --device cuda --out outputs/fused_IN1_gpt2-medium.csv
$PY sources/lm_calib.py  --tags $S --mix outputs/fused_IN1_llm_plus_lmmix.npy outputs/fused_IN1_gpt2-medium_lmmix.npy --alpha 0.5 --out outputs/fused_IN1_Q_llm_plus.csv
$PY sources/make_final.py --fused outputs/fused_IN1_Q_llm_plus.csv --tags $S --iso_tags $ISO --tau 0.7 --out outputs/final_IN1_Q_llm_plus.csv

# public 0.733: whole-sentence candidates, LM prior 0.5, lambda 0.3
$PY sources/sent_cand.py --tags $S --corpus outputs/vocab_corpus_llm_big.txt outputs/vocab_corpus_llm_long.txt outputs/vocab_corpus_guided.txt \
    --base outputs/final_IN1_Q_llm_plus.csv --lm outputs/lm_gpt2_llm_plus --beta 0.5 --T 1 --lam 0.3 --out outputs/final_IN1Qp_SCGL_b0.5.csv
```

`sent_cand.py` mixes arithmetically and puts uniform mass 0.02 inside the candidate marginals (defaults). A sentence keeps the base prediction when fewer than 20 candidates of its length exist; with these pools that applies to 4 of the 156 test sentences. With our inputs, running `sent_cand.py` again reproduced `final_IN1Qp_SCGL_b0.5.csv` exactly (maximum absolute difference 0.0). The ablations in the README vary `--corpus`, `--beta` and `--meg_w` (`--meg_w 0` drops the MEG term).

To generate new guided proposals instead of using the released pool (LLM sampling gives a different pool each run):
```bash
$PY sources/gen_guided.py --sub outputs/final_IN1_Q_llm_plus.csv --k 6 --n_per 3 --out outputs/vocab_corpus_guided_new.txt
```

## 5. Checkpoints

The checkpoints are not stored in git. They are shared from Google Drive: https://drive.google.com/drive/folders/1HIGSl6Yx3K_mzwb7Op_ZBc0fdD9w8FNT (the folder also contains checkpoints of development runs that are not part of the final ensembles).
- `<tag>/model.pt` and `<tag>/holdout_logits.npz` for every Colab member listed above.
- `release_extras/local_members.zip`: the members trained on a local GPU (`f_s1-3`, `fe_s4`, `pfl_s1-3`).
- `release_extras/lm_gpt2_llm_plus.zip` and `release_extras/lm_gpt2_llm_big.zip`: the fine-tuned LMs.

Place each `<tag>` folder under `outputs/`. With the checkpoints and `holdout_logits.npz` in place, Section 4 reproduces the submissions without retraining.

Some folders also contain `metrics.json` (training history), but not those of `f7_s1-3` and of the self-trained members. Every checkpoint stores its arguments:
```bash
$PY -c "import torch, sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['args'])" outputs/<tag>/model.pt
```
For a self-trained member these are the arguments of the model it started from, plus `pseudo_from` and `teacher`; its self-training settings are the ones in the table in Section 2.

## 6. Analysis and write-up

From the repository root:
```bash
cd analysis
python analyze_lb.py     # data/analysis.json, data/milestones.json, ../paper/fig_*.pdf, previews data/fig_*.png
python make_tables.py    # ../paper/tab_pairs_main.tex, ../paper/tab_pairs_all.tex
cd ../paper
tectonic main.tex        # main.pdf (released as paper.pdf)
```

Inputs in `analysis/data/`:
- `submissions_raw.csv`: Kaggle export of all 99 scored submissions with public and private scores (`kaggle competitions submissions pnpl-competition-2026-deep --csv --page-size 200`), made after the private leaderboard was released. A new export on 29 October 2026 gave the same file names and scores.
- `public_leaderboard_2026-10-29.csv` (all 91 teams) and `private_leaderboard_top49_2026-10-29.csv` (the first 49 places): leaderboard snapshots saved on 29 October 2026, used only for the ranks. Team names other than ours are removed.
- `iso_check.json`: for the pairs of submissions that differ only in the isolated-word rows, the largest difference on the sentence rows and the number of isolated rows whose top-10 set differs. It was written by `python iso_check.py ../experiments/20261008_full_subj0_deepnet/outputs` from our submission files; the files of one pair were not kept.

`analyze_lb.py` computes every leaderboard statistic from these files. The following inputs are written into the script by hand, from our dated experiment log (not released): the 30 comparisons with their single or bundle label, the validation direction of each comparison with its date and validation scores, the pairs that differ only in isolated-word rows, and the two final selections (actual: chosen on 16 October; one day earlier: recorded on 15 October). The script's results are therefore conditional on these annotations.
