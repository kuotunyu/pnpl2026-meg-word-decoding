#!/bin/bash
# Colab 10/15 night (~1.5 h, ONE job per 40 GB GPU): 3 s full-data member with per-window demean + one global scale per window, seed 62.
# Motivation: arXiv 2604.04129 (LibriBrain) finds normalisation-induced distribution shift is the main
# val->test gap and instance normalisation removes it; our holdout windows are ~1.35x larger than training ones.
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day27b.log"; }
log "start"
python -u safe_copy.py "$DRV/derived_w3/subj0_pca_w3" "$DER/subj0_pca_w3" 2>&1 | tail -1 | tee -a "$DRV/logs/day27b.log"
df -h /content | tail -1 | tee -a "$DRV/logs/day27b.log"
timeout 120 python - <<'PY' || echo 'HF download skipped (rate limit); using sources/holdout_wordmask.csv'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
train() { tag=$1; shift
  python -u train_ctx.py --tag $tag --data subj0_pca_w3 --full --w3aug --epochs 40 --ema 0.999 "$@" > "$DRV/logs/$tag.log" 2>&1
  cp -r "$OUT/$tag" "$DRV/outputs/"
  log "$tag finished: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-120) holdout=$(ls "$OUT/$tag/holdout_logits.npz" 2>/dev/null | wc -l)"; }
train w3G_s62 --seed 62 --layers 8 --prep gnorm
log "DAY27B_DONE"
