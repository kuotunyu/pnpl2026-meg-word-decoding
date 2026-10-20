#!/bin/bash
# Colab run 10/12 (second runtime, ~2.5 h): self-training with teacher final_B2 (sources/teacher_B2.csv) of the
# Colab-trained ORIGINAL diverse members -> t* tags:
#  w3f_s1..s8, w3ef_s12, w3df_s13 (3 s 50 Hz)  ->  tw3_s1..s8, tw3e_s12, tw3d_s13
#  f7_s1..s3, f7e_s4, f7e_s5 (1 s)              ->  tf7_s1..s3, tf7e_s4, tf7e_s5
#  w3hf_s1..s3 (3 s 100 Hz)                     ->  tw3h_s1..s3
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day12.log"; }
log "start"
[ -d "$DER/subj0_pca_w3/runs" ] || cp -r "$DRV/derived_w3/subj0_pca_w3" "$DER/"
[ -d "$DER/subj0_pca_epochs/runs" ] || cp -r "$DRV/derived/subj0_pca_epochs" "$DER/"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
[ -f "$DER/subj0_pca_epochs/holdout_subj00.npz" ] || python -u preprocess.py holdout 2>&1 | tail -1 | tee -a "$DRV/logs/day12.log"
pip -q install hf_xet 2>&1 | tail -1
for t in w3f_s1 w3f_s2 w3f_s3 w3f_s4 w3f_s5 w3f_s6 w3f_s7 w3f_s8 w3ef_s12 w3df_s13 f7_s1 f7_s2 f7_s3 f7e_s4 f7e_s5 w3hf_s1 w3hf_s2 w3hf_s3; do
  mkdir -p "$OUT/$t"; cp "$DRV/outputs/$t/model.pt" "$OUT/$t/"; done
log "models copied"
ft() { tag=$1; fs=$2; shift 2
  PNPL_FS=$fs python -u pseudo_ft.py --tag $tag --teacher teacher_B2.csv "$@" > "$DRV/logs/$tag.log" 2>&1
  mkdir -p "$DRV/outputs/$tag"; cp "$OUT/$tag/holdout_logits.npz" "$OUT/$tag/model.pt" "$DRV/outputs/$tag/" 2>/dev/null
  log "$tag: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-150)"; }
( for s in 1 2 3 4 5; do ft tw3_s$s 50 --init w3f_s$s --data subj0_pca_w3 --w3aug --seed $((60+s)); done ) &
sleep 90
( for s in 6 7 8; do ft tw3_s$s 50 --init w3f_s$s --data subj0_pca_w3 --w3aug --seed $((60+s)); done
  ft tw3e_s12 50 --init w3ef_s12 --data subj0_pca_w3 --w3aug --seed 72
  ft tw3d_s13 50 --init w3df_s13 --data subj0_pca_w3 --w3aug --seed 73 ) &
sleep 90
( for s in 1 2 3; do ft tf7_s$s 125 --init f7_s$s --data subj0_pca_epochs --seed $((80+s)); done
  ft tf7e_s4 125 --init f7e_s4 --data subj0_pca_epochs --seed 84
  ft tf7e_s5 125 --init f7e_s5 --data subj0_pca_epochs --seed 85 ) &
wait
python -u download_subj0.py > "$DRV/logs/day12_download.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py runs 10 > "$DRV/logs/day12_w3hprep.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py holdout >> "$DRV/logs/day12_w3hprep.log" 2>&1
rm -rf /content/pnpl/data/raw/LibriBrain /content/pnpl/data/raw/LibriBrain2
for s in 1 2 3; do ft tw3h_s$s 100 --init w3hf_s$s --data subj0_pca_w3h --w3aug --seed $((90+s)); done
log "DAY12_DONE"
