#!/bin/bash
# day23 (10/14 early morning, ~2.5-3 h): the official reference depth. 16-layer d1024 at lr 1e-4 (12-layer needed lr <= 2e-4;
# w3M_s41 / w3N_s43 then reached loss 8.42 / 8.45 vs 8.58 for the 8-layer family). One job per 40 GB GPU.
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day23.log"; }
log "start"
python -u safe_copy.py "$DRV/derived_w3/subj0_pca_w3" "$DER/subj0_pca_w3" 2>&1 | tail -1 | tee -a "$DRV/logs/day23.log"
df -h /content | tail -1 | tee -a "$DRV/logs/day23.log"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
train() { tag=$1; shift
  python -u train_ctx.py --tag $tag --data subj0_pca_w3 --full --w3aug --epochs 40 --ema 0.999 "$@" > "$DRV/logs/$tag.log" 2>&1
  cp -r "$OUT/$tag" "$DRV/outputs/"
  log "$tag finished: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-120) holdout=$(ls "$OUT/$tag/holdout_logits.npz" 2>/dev/null | wc -l)"; }
train w3X_s51 --seed 51 --layers 16 --d 1024 --lr 1e-4
log "DAY23_DONE"
