#!/bin/bash
# Colab run 10/10 evening (~3.5 h):
#  r6_dm   : validation, 3 s window 50 Hz + per-window mean removal (--prep demean; d'Ascoli 2025 baseline correction)
#            reference r4_ema: novel 0.6526, proxy tau0/0.3/0.5 0.530/0.557/0.576, iso tau0.7 0.451
#  w3hf_s1 : full-data model, 3 s window at 100 Hz (r5_h100 validation: novel 0.648, proxy 0.563/0.599/0.600, iso 0.442)
#            -> to test the 100 Hz model inside the ensemble on Public
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day4.log"; }
log "start"
[ -d "$DER/subj0_pca_w3/runs" ] || cp -r "$DRV/derived_w3/subj0_pca_w3" "$DER/"
mkdir -p "$DER/subj0_pca_epochs" && cp "$DRV/derived/subj0_pca_epochs/pca_W.npy" "$DER/subj0_pca_epochs/"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
pip -q install hf_xet 2>&1 | tail -1
run() { tag=$1; fs=$2; data=$3; shift 3
  ( PNPL_FS=$fs python -u train_ctx.py --tag $tag --data $data "$@" > "$DRV/logs/$tag.log" 2>&1
    cp -r "$OUT/$tag" "$DRV/outputs/"
    log "$tag finished: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-170)"
    if [ -f "$OUT/$tag/val_logits.npz" ]; then
      PNPL_FS=$fs python -u iso_eval.py $tag cuda $data 2>&1 | grep iso | tee -a "$DRV/logs/day4.log"
      PNPL_FS=$fs python -u proxy_eval.py $tag cuda $data 2>&1 | grep proxy | tee -a "$DRV/logs/day4.log"
      cp "$OUT/$tag/iso_val.npz" "$OUT/$tag/proxy_logits.npz" "$DRV/outputs/$tag/" 2>/dev/null
    fi ) &
  sleep 150; }
run r6_dm 50 subj0_pca_w3 --w3aug --layers 8 --epochs 40 --ema 0.999 --prep demean
python -u download_subj0.py > "$DRV/logs/day4_download.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py runs 10 > "$DRV/logs/day4_w3hprep.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py holdout >> "$DRV/logs/day4_w3hprep.log" 2>&1
rm -rf /content/pnpl/data/raw/LibriBrain /content/pnpl/data/raw/LibriBrain2
log "w3h data: $(ls $DER/subj0_pca_w3h/runs | wc -l) runs; $(tail -1 "$DRV/logs/day4_w3hprep.log")"
run w3hf_s1 100 subj0_pca_w3h --full --seed 1 --w3aug --layers 8 --epochs 40 --ema 0.999
wait
log "DAY4_DONE"
