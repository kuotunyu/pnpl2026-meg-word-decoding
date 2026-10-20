#!/bin/bash
# Colab run 10/12 (~1.5 h): self-training with the stronger teacher final_B2 (sources/teacher_B2.csv, Public 0.69834)
# of the day10 originals: w3e_s14, w3e_s15 (50 Hz EMA), f7e_s6 (1 s), w3hf_s4 (100 Hz) -> sw3e_s14, sw3e_s15, sf7e_s6, sw3h_s4
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day11.log"; }
log "start"
[ -d "$DER/subj0_pca_w3/runs" ] || cp -r "$DRV/derived_w3/subj0_pca_w3" "$DER/"
[ -d "$DER/subj0_pca_epochs/runs" ] || cp -r "$DRV/derived/subj0_pca_epochs" "$DER/"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
[ -f "$DER/subj0_pca_epochs/holdout_subj00.npz" ] || python -u preprocess.py holdout 2>&1 | tail -1 | tee -a "$DRV/logs/day11.log"
pip -q install hf_xet 2>&1 | tail -1
for t in w3e_s14 w3e_s15 f7e_s6 w3hf_s4; do mkdir -p "$OUT/$t"; cp "$DRV/outputs/$t/model.pt" "$OUT/$t/"; done
log "models copied"
ft() { tag=$1; fs=$2; shift 2
  PNPL_FS=$fs python -u pseudo_ft.py --tag $tag --teacher teacher_B2.csv "$@" > "$DRV/logs/$tag.log" 2>&1
  mkdir -p "$DRV/outputs/$tag"; cp "$OUT/$tag/holdout_logits.npz" "$OUT/$tag/model.pt" "$DRV/outputs/$tag/" 2>/dev/null
  log "$tag: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-150)"; }
( ft sw3e_s14 50 --init w3e_s14 --data subj0_pca_w3 --w3aug --seed 41 ) &
sleep 90
( ft sw3e_s15 50 --init w3e_s15 --data subj0_pca_w3 --w3aug --seed 42 ) &
sleep 90
( ft sf7e_s6 125 --init f7e_s6 --data subj0_pca_epochs --seed 43 ) &
wait
python -u download_subj0.py > "$DRV/logs/day11_download.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py runs 10 > "$DRV/logs/day11_w3hprep.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py holdout >> "$DRV/logs/day11_w3hprep.log" 2>&1
rm -rf /content/pnpl/data/raw/LibriBrain /content/pnpl/data/raw/LibriBrain2
ft sw3h_s4 100 --init w3hf_s4 --data subj0_pca_w3h --w3aug --seed 44
log "DAY11_DONE"
