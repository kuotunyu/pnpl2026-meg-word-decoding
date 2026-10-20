#!/bin/bash
# Colab 10/13-14 night (~5-6 h): retry the DEEPER transformer with a learning rate suited to its size.
# 12-layer d768 at the shared lr 5e-4 trained badly (w3L final loss 8.85-9.45 vs 8.57 for w3f; validation rL_s1 never
# learned). Official reference (d'Ascoli 2025) is 16-layer and scores 73.2 % on the competition test splits.
#  w3M_s41: 12-layer d768 lr 2e-4    w3M_s42: same, seed 42    w3N_s43: 12-layer d768 lr 1.5e-4
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day21.log"; }
log "start"
python -u safe_copy.py "$DRV/derived_w3/subj0_pca_w3" "$DER/subj0_pca_w3" 2>&1 | tail -1 | tee -a "$DRV/logs/day21.log"
df -h /content | tail -1 | tee -a "$DRV/logs/day21.log"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
train() { tag=$1; shift
  python -u train_ctx.py --tag $tag --data subj0_pca_w3 --full --w3aug --epochs 40 --ema 0.999 "$@" > "$DRV/logs/$tag.log" 2>&1
  cp -r "$OUT/$tag" "$DRV/outputs/"
  log "$tag finished: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-120) holdout=$(ls "$OUT/$tag/holdout_logits.npz" 2>/dev/null | wc -l)"; }
( train w3M_s41 --seed 41 --layers 12 --d 768 --lr 2e-4 ) &
sleep 150
( train w3M_s42 --seed 42 --layers 12 --d 768 --lr 2e-4 ) &
sleep 150
( train w3N_s43 --seed 43 --layers 12 --d 768 --lr 1.5e-4 ) &
wait
log "DAY21_DONE"
