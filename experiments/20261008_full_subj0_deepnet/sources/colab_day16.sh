#!/bin/bash
# Colab second runtime 10/13 night (~4.5 h, runs in parallel with day15): more full-data ARCHITECTURE-diverse 3 s members
# (w3L_s25, w3L_s28: 12-layer d768; w3C_s27: CNN width 512). Used in the final broad ensemble only if day15
# validation confirms that architecture diversity helps.
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day16.log"; }
log "start"
python -u safe_copy.py "$DRV/derived_w3/subj0_pca_w3" "$DER/subj0_pca_w3" 2>&1 | tail -1 | tee -a "$DRV/logs/day16.log"
df -h /content | tail -1 | tee -a "$DRV/logs/day16.log"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
train() { tag=$1; shift
  python -u train_ctx.py --tag $tag --data subj0_pca_w3 --full --w3aug --epochs 40 --ema 0.999 "$@" > "$DRV/logs/$tag.log" 2>&1
  cp -r "$OUT/$tag" "$DRV/outputs/"
  log "$tag finished: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-120) holdout=$(ls "$OUT/$tag/holdout_logits.npz" 2>/dev/null | wc -l)"; }
( train w3L_s25 --seed 25 --layers 12 --d 768 ) &
sleep 150
( train w3C_s27 --seed 27 --layers 8 --width 512 ) &
sleep 150
( train w3L_s28 --seed 28 --layers 12 --d 768 ) &
wait
log "DAY16_DONE"
