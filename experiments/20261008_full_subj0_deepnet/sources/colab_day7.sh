#!/bin/bash
# Colab run 10/11 evening (~1.5 h): self-training on the unlabeled holdout (pseudo_ft.py).
# Teacher = accepted submission finalW_calib2_iso14 (sources/teacher_W.csv, Public 0.67925).
# Fine-tunes 4 members of the accepted ensemble: w3f_s1, w3f_s2, w3f_s3 (3 s, 50 Hz) and f7_s1 (1 s) -> pw3_s1..3, pf7_s1
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day7.log"; }
log "start"
[ -d "$DER/subj0_pca_w3/runs" ] || cp -r "$DRV/derived_w3/subj0_pca_w3" "$DER/"
[ -d "$DER/subj0_pca_epochs/runs" ] || cp -r "$DRV/derived/subj0_pca_epochs" "$DER/"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
[ -f "$DER/subj0_pca_epochs/holdout_subj00.npz" ] || python -u preprocess.py holdout 2>&1 | tail -1 | tee -a "$DRV/logs/day7.log"
for t in w3f_s1 w3f_s2 w3f_s3 f7_s1; do mkdir -p "$OUT/$t"; cp "$DRV/outputs/$t/model.pt" "$OUT/$t/"; done
log "data ready"
ft() { tag=$1; shift
  python -u pseudo_ft.py --tag $tag --teacher teacher_W.csv "$@" > "$DRV/logs/$tag.log" 2>&1
  mkdir -p "$DRV/outputs/$tag"; cp "$OUT/$tag/holdout_logits.npz" "$OUT/$tag/model.pt" "$DRV/outputs/$tag/" 2>/dev/null
  log "$tag: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-150)"; }
( ft pw3_s1 --init w3f_s1 --data subj0_pca_w3 --w3aug --seed 1
  ft pw3_s3 --init w3f_s3 --data subj0_pca_w3 --w3aug --seed 3 ) &
sleep 120
( ft pw3_s2 --init w3f_s2 --data subj0_pca_w3 --w3aug --seed 2
  ft pf7_s1 --init f7_s1 --data subj0_pca_epochs --seed 4 ) &
wait
log "DAY7_DONE"
