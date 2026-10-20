#!/bin/bash
# Colab run 10/11 night (~1.5 h): self-training round 2. Teacher = final_P1 (sources/teacher_P1.csv, Public 0.69430).
# Fine-tunes ALL 9 members of the accepted ensemble from their original weights -> q* tags.
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day8.log"; }
log "start"
[ -d "$DER/subj0_pca_w3/runs" ] || cp -r "$DRV/derived_w3/subj0_pca_w3" "$DER/"
[ -d "$DER/subj0_pca_epochs/runs" ] || cp -r "$DRV/derived/subj0_pca_epochs" "$DER/"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
[ -f "$DER/subj0_pca_epochs/holdout_subj00.npz" ] || python -u preprocess.py holdout 2>&1 | tail -1 | tee -a "$DRV/logs/day8.log"
for t in w3f_s1 w3f_s2 w3f_s3 w3f_s4 w3f_s5 w3f_s6 f7_s1 f7_s2 f7_s3; do mkdir -p "$OUT/$t"; cp "$DRV/outputs/$t/model.pt" "$OUT/$t/"; done
log "data ready"
ft() { tag=$1; shift
  python -u pseudo_ft.py --tag $tag --teacher teacher_P1.csv "$@" > "$DRV/logs/$tag.log" 2>&1
  mkdir -p "$DRV/outputs/$tag"; cp "$OUT/$tag/holdout_logits.npz" "$OUT/$tag/model.pt" "$DRV/outputs/$tag/" 2>/dev/null
  log "$tag: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-150)"; }
( for s in 1 2 3; do ft qw3_s$s --init w3f_s$s --data subj0_pca_w3 --w3aug --seed $s; done ) &
sleep 90
( for s in 4 5 6; do ft qw3_s$s --init w3f_s$s --data subj0_pca_w3 --w3aug --seed $s; done ) &
sleep 90
( for s in 1 2 3; do ft qf7_s$s --init f7_s$s --data subj0_pca_epochs --seed $((s+10)); done ) &
wait
log "DAY8_DONE"
