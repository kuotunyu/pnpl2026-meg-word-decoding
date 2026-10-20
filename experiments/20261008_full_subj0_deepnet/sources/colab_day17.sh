#!/bin/bash
# Colab 10/13 day (~4.5 h): more WIDE-CNN full-data 3 s members. day15 validation: CNN width 512 is the best single
# model (proxy 0.610 vs 0.567 same-setting seed) and adding it helps proxy +0.020 (P 0.98) and iso +0.007 (P 0.93);
# demean helps proxy +0.022. 12-layer d768 is unstable (rL_s1 failed) -> dropped.
#  w3C_s29, w3C_s31: CNN width 512      w3Cd_s30: CNN width 512 + per-window mean removal
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day17.log"; }
log "start"
python -u safe_copy.py "$DRV/derived_w3/subj0_pca_w3" "$DER/subj0_pca_w3" 2>&1 | tail -1 | tee -a "$DRV/logs/day17.log"
df -h /content | tail -1 | tee -a "$DRV/logs/day17.log"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
train() { tag=$1; shift
  python -u train_ctx.py --tag $tag --data subj0_pca_w3 --full --w3aug --epochs 40 --ema 0.999 "$@" > "$DRV/logs/$tag.log" 2>&1
  cp -r "$OUT/$tag" "$DRV/outputs/"
  log "$tag finished: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-120) holdout=$(ls "$OUT/$tag/holdout_logits.npz" 2>/dev/null | wc -l)"; }
( train w3C_s29 --seed 29 --layers 8 --width 512 ) &
sleep 150
( train w3Cd_s30 --seed 30 --layers 8 --width 512 --prep demean ) &
sleep 150
( train w3C_s31 --seed 31 --layers 8 --width 512 ) &
wait
log "DAY16_DONE"
