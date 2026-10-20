#!/bin/bash
# Colab (A100 high-RAM): rebuild subj0 PCA epochs from Hugging Face and save them to Drive.
# Expects this sources folder at /content/pnpl/experiments/20261008_full_subj0_deepnet/sources and Drive mounted.
set -e
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
LOG="$DRV/colab_prep.log"
mkdir -p "$DRV"
cd "$SRC"
echo "start $(date -u)" | tee -a "$LOG"
pip -q install hf_xet h5py 2>&1 | tail -1
python -u download_subj0.py 2>&1 | grep -E "files|DONE|FAIL" | tee -a "$LOG"
python -u - <<'PY' 2>&1 | tee -a "$LOG"
from huggingface_hub import hf_hub_download, snapshot_download
root = "/content/pnpl/data/raw"
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir=root + "/LibriBrain-Competition-2026")
snapshot_download("google-t5/t5-large", allow_patterns=["config.json", "model.safetensors", "spiece.model"],
                  local_dir=root + "/google-t5_t5-large_150ebc2c4b72291e770f58e6057481c8d2ed331a")
print("holdout + t5 downloaded")
PY
python -u preprocess.py pca 2>&1 | tail -1 | tee -a "$LOG"
python -u preprocess.py runs 8 2>&1 | tail -1 | tee -a "$LOG"
python -u preprocess.py holdout 2>&1 | tail -1 | tee -a "$LOG"
python -u t5_embed.py 2>&1 | tail -1 | tee -a "$LOG"
echo "runs: $(ls /content/pnpl/data/derived/subj0_pca_epochs/runs | wc -l)" | tee -a "$LOG"
mkdir -p "$DRV/derived"
cp -r /content/pnpl/data/derived/* "$DRV/derived/"
echo "PREP_DONE copied to Drive $(date -u)" | tee -a "$LOG"
