#!/bin/bash
# Colab night run 10/12 (~6 h, second runtime): more diverse full-data members, all with EMA.
#  slot A: w3hf_s4 (3 s 100 Hz) -> f7e_s6 (1 s)
#  slot B: w3e_s14 (3 s 50 Hz, EMA) -> w3e_s15
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day10.log"; }
log "start"
[ -d "$DER/subj0_pca_w3/runs" ] || cp -r "$DRV/derived_w3/subj0_pca_w3" "$DER/"
[ -d "$DER/subj0_pca_epochs/runs" ] || cp -r "$DRV/derived/subj0_pca_epochs" "$DER/"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
[ -f "$DER/subj0_pca_epochs/holdout_subj00.npz" ] || python -u preprocess.py holdout 2>&1 | tail -1 | tee -a "$DRV/logs/day10.log"
pip -q install hf_xet 2>&1 | tail -1
train() { tag=$1; fs=$2; data=$3; shift 3
  PNPL_FS=$fs python -u train_ctx.py --tag $tag --data $data "$@" > "$DRV/logs/$tag.log" 2>&1
  cp -r "$OUT/$tag" "$DRV/outputs/"
  log "$tag finished: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-120) holdout=$(ls "$OUT/$tag/holdout_logits.npz" 2>/dev/null | wc -l)"; }
( train w3e_s14 50 subj0_pca_w3 --full --seed 14 --w3aug --layers 8 --epochs 40 --ema 0.999
  train w3e_s15 50 subj0_pca_w3 --full --seed 15 --w3aug --layers 8 --epochs 40 --ema 0.999 ) &
python -u download_subj0.py > "$DRV/logs/day10_download.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py runs 10 > "$DRV/logs/day10_w3hprep.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py holdout >> "$DRV/logs/day10_w3hprep.log" 2>&1
rm -rf /content/pnpl/data/raw/LibriBrain /content/pnpl/data/raw/LibriBrain2
log "w3h data: $(ls $DER/subj0_pca_w3h/runs | wc -l) runs"
train w3hf_s4 100 subj0_pca_w3h --full --seed 4 --w3aug --layers 8 --epochs 40 --ema 0.999
rm -rf "$DER/subj0_pca_w3h"
train f7e_s6 125 subj0_pca_epochs --full --seed 6 --layers 8 --epochs 40 --ema 0.999
wait
log "DAY10_DONE"
