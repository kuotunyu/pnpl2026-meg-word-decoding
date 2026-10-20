#!/bin/bash
# Colab run 10/11 (~4.5 h): more DIVERSE full-data ensemble members (validation 10/11: mixing different settings
# adds +0.013-0.019 novel / +0.01-0.02 isolated-word; e.g. 50 Hz + 100 Hz 0.6697 vs 0.6526 alone).
#  slot 1: w3hf_s2  (3 s window, 100 Hz, EMA)
#  slot 2: f7e_s4   (1 s window c7 setting, EMA)  then  w3vf_s1 (3 s 50 Hz, vocab-only context 30 %, EMA)
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day5.log"; }
log "start"
[ -d "$DER/subj0_pca_w3/runs" ] || cp -r "$DRV/derived_w3/subj0_pca_w3" "$DER/"
[ -d "$DER/subj0_pca_epochs/runs" ] || cp -r "$DRV/derived/subj0_pca_epochs" "$DER/"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
[ -f "$DER/subj0_pca_epochs/holdout_subj00.npz" ] || python -u preprocess.py holdout 2>&1 | tail -1 | tee -a "$DRV/logs/day5.log"
pip -q install hf_xet 2>&1 | tail -1
train() { tag=$1; fs=$2; data=$3; shift 3
  PNPL_FS=$fs python -u train_ctx.py --tag $tag --data $data "$@" > "$DRV/logs/$tag.log" 2>&1
  cp -r "$OUT/$tag" "$DRV/outputs/"
  log "$tag finished: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-120) holdout=$(ls "$OUT/$tag/holdout_logits.npz" 2>/dev/null | wc -l)"; }
# slot 2 starts at once (data already on disk)
( train f7e_s4 125 subj0_pca_epochs --full --seed 4 --layers 8 --epochs 40 --ema 0.999
  train w3vf_s1 50 subj0_pca_w3 --full --seed 11 --w3aug --layers 8 --epochs 40 --ema 0.999 --vocabctx 0.3 ) &
# slot 1 needs the 100 Hz data (rebuilt from HF, ~20 min)
python -u download_subj0.py > "$DRV/logs/day5_download.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py runs 10 > "$DRV/logs/day5_w3hprep.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py holdout >> "$DRV/logs/day5_w3hprep.log" 2>&1
rm -rf /content/pnpl/data/raw/LibriBrain /content/pnpl/data/raw/LibriBrain2
log "w3h data: $(ls $DER/subj0_pca_w3h/runs | wc -l) runs"
train w3hf_s2 100 subj0_pca_w3h --full --seed 2 --w3aug --layers 8 --epochs 40 --ema 0.999
wait
log "DAY5_DONE"
