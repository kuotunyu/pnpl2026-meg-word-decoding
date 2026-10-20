#!/bin/bash
# Colab run 10/12 (~1.5-2 h): self-training (teacher = final_P1, sources/teacher_P1.csv) of the DIVERSE members
# not yet self-trained: w3f_s7/s8, w3ef_s12 (60 ep), w3df_s13 (demean), f7e_s4/s5 (1 s), w3hf_s1..s3 (100 Hz) -> r* tags
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day9.log"; }
log "start"
[ -d "$DER/subj0_pca_w3/runs" ] || cp -r "$DRV/derived_w3/subj0_pca_w3" "$DER/"
[ -d "$DER/subj0_pca_epochs/runs" ] || cp -r "$DRV/derived/subj0_pca_epochs" "$DER/"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
[ -f "$DER/subj0_pca_epochs/holdout_subj00.npz" ] || python -u preprocess.py holdout 2>&1 | tail -1 | tee -a "$DRV/logs/day9.log"
pip -q install hf_xet 2>&1 | tail -1
for t in w3f_s7 w3f_s8 w3ef_s12 w3df_s13 f7e_s4 f7e_s5 w3hf_s1 w3hf_s2 w3hf_s3; do mkdir -p "$OUT/$t"; cp "$DRV/outputs/$t/model.pt" "$OUT/$t/"; done
log "models copied"
ft() { tag=$1; fs=$2; shift 2
  PNPL_FS=$fs python -u pseudo_ft.py --tag $tag --teacher teacher_P1.csv "$@" > "$DRV/logs/$tag.log" 2>&1
  mkdir -p "$DRV/outputs/$tag"; cp "$OUT/$tag/holdout_logits.npz" "$OUT/$tag/model.pt" "$DRV/outputs/$tag/" 2>/dev/null
  log "$tag: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-150)"; }
( ft rw3_s7 50 --init w3f_s7 --data subj0_pca_w3 --w3aug --seed 7
  ft rw3e_s12 50 --init w3ef_s12 --data subj0_pca_w3 --w3aug --seed 12 ) &
sleep 90
( ft rw3_s8 50 --init w3f_s8 --data subj0_pca_w3 --w3aug --seed 8
  ft rw3d_s13 50 --init w3df_s13 --data subj0_pca_w3 --w3aug --seed 13 ) &
sleep 90
( ft rf7e_s4 125 --init f7e_s4 --data subj0_pca_epochs --seed 14
  ft rf7e_s5 125 --init f7e_s5 --data subj0_pca_epochs --seed 15 ) &
wait
python -u download_subj0.py > "$DRV/logs/day9_download.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py runs 10 > "$DRV/logs/day9_w3hprep.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py holdout >> "$DRV/logs/day9_w3hprep.log" 2>&1
rm -rf /content/pnpl/data/raw/LibriBrain /content/pnpl/data/raw/LibriBrain2
log "w3h data: $(ls $DER/subj0_pca_w3h/runs | wc -l) runs"
for s in 1 2 3; do ft rw3h_s$s 100 --init w3hf_s$s --data subj0_pca_w3h --w3aug --seed $((20+s)); done
log "DAY9_DONE"
