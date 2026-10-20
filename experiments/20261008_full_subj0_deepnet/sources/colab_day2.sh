#!/bin/bash
# Colab day run 10/10 (~3 h): training-side changes for the balanced (per-class) metric, 3 validation runs at once
#  r4_bal6 / r4_bal10: class weights prior^-0.6 / prior^-1.0 in the loss (current 0.3)
#  r4_ema: exponential moving average of the weights (decay 0.999)
#  each: normal + novel validation, isolated-word validation, holdout-like (proxy) validation -> Drive outputs/<tag>/
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day2.log"; }
log "start"
[ -d "$DER/subj0_pca_w3/runs" ] || cp -r "$DRV/derived_w3/subj0_pca_w3" "$DER/"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
log "data: w3 $(ls $DER/subj0_pca_w3/runs | wc -l) runs"

evals() { tag=$1
  python -u iso_eval.py $tag cuda subj0_pca_w3 2>&1 | grep iso | tee -a "$DRV/logs/day2.log"
  python -u proxy_eval.py $tag cuda subj0_pca_w3 2>&1 | grep proxy | tee -a "$DRV/logs/day2.log"
  cp "$OUT/$tag/iso_val.npz" "$OUT/$tag/proxy_logits.npz" "$DRV/outputs/$tag/" 2>/dev/null; }
run() { tag=$1; shift
  ( python -u train_ctx.py --tag $tag "$@" > "$DRV/logs/$tag.log" 2>&1
    cp -r "$OUT/$tag" "$DRV/outputs/"
    log "$tag finished: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-170)"
    evals $tag ) &
  sleep 150; }
BASE_ARGS="--data subj0_pca_w3 --w3aug --layers 8 --epochs 40"
run r4_bal6 $BASE_ARGS --wpow 0.6
run r4_bal10 $BASE_ARGS --wpow 1.0
run r4_ema $BASE_ARGS --ema 0.999
wait
log "DAY2_DONE (reference r1_w3: novel 0.6505, proxy tau0/0.3/0.5 0.525/0.557/0.576, iso tau0.7 0.445)"
