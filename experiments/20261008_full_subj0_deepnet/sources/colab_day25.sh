#!/bin/bash
# Colab 10/14 early morning (~2.5 h, jobs one after another for a 40 GB GPU):
#  1) self-training (teacher final_B5i) of the now properly trained 12-layer models -> uw3M_s41, uw3N_s43
#  2) f7M_s46: 1 s window, 12-layer d768, lr 2e-4 (the lr fix applied to the 1 s family)
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day25.log"; }
log "start"
# verified copies (10/12 first try: plain cp -r gave truncated .npz files -> every job failed with EOFError)
python -u safe_copy.py "$DRV/derived_w3/subj0_pca_w3" "$DER/subj0_pca_w3" 2>&1 | tail -1 | tee -a "$DRV/logs/day25.log"
python -u safe_copy.py "$DRV/derived/subj0_pca_epochs" "$DER/subj0_pca_epochs" 2>&1 | tail -1 | tee -a "$DRV/logs/day25.log"
df -h /content | tail -1 | tee -a "$DRV/logs/day25.log"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
[ -f "$DER/subj0_pca_epochs/holdout_subj00.npz" ] || python -u preprocess.py holdout 2>&1 | tail -1 | tee -a "$DRV/logs/day25.log"
for t in w3M_s41 w3N_s43; do mkdir -p "$OUT/$t"; cp "$DRV/outputs/$t/model.pt" "$OUT/$t/"; ls -l "$OUT/$t/model.pt" "$DRV/outputs/$t/model.pt" | tee -a "$DRV/logs/day25.log"; done
train() { tag=$1; data=$2; shift 2
  python -u train_ctx.py --tag $tag --data $data --full --epochs 40 --ema 0.999 "$@" > "$DRV/logs/$tag.log" 2>&1
  cp -r "$OUT/$tag" "$DRV/outputs/"
  log "$tag finished: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-120) holdout=$(ls "$OUT/$tag/holdout_logits.npz" 2>/dev/null | wc -l)"; }
ft() { tag=$1; shift
  python -u pseudo_ft.py --tag $tag --teacher teacher_B5i.csv "$@" > "$DRV/logs/$tag.log" 2>&1
  mkdir -p "$DRV/outputs/$tag"; cp "$OUT/$tag/holdout_logits.npz" "$OUT/$tag/model.pt" "$DRV/outputs/$tag/" 2>/dev/null
  log "$tag: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-150)"; }
train() { tag=$1; data=$2; shift 2
  python -u train_ctx.py --tag $tag --data $data --full --epochs 40 --ema 0.999 "$@" > "$DRV/logs/$tag.log" 2>&1
  cp -r "$OUT/$tag" "$DRV/outputs/"
  log "$tag finished: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-120) holdout=$(ls "$OUT/$tag/holdout_logits.npz" 2>/dev/null | wc -l)"; }
ft uw3M_s41 --init w3M_s41 --data subj0_pca_w3 --w3aug --seed 141
ft uw3N_s43 --init w3N_s43 --data subj0_pca_w3 --w3aug --seed 143
train f7M_s46 subj0_pca_epochs --seed 46 --layers 12 --d 768 --lr 2e-4
log "DAY25_DONE"
