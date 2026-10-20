#!/bin/bash
# Colab run 10/10 (~4-5 h): two validation runs on the 3 s window (the Public-best component), both with weight EMA.
#  r5_h100: 3 s window at 100 Hz instead of 50 Hz (data rebuilt from HF: subj0_pca_w3h)
#  r5_ep60: 3 s window at 50 Hz, 60 epochs instead of 40
#  reference r4_ema (50 Hz, 40 ep, EMA): novel 0.6526, proxy tau0/0.3/0.5 0.530/0.557/0.576, iso tau0.7 0.451
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day3.log"; }
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
    PNPL_FS=$fs python -u iso_eval.py $tag cuda $data 2>&1 | grep iso | tee -a "$DRV/logs/day3.log"
    PNPL_FS=$fs python -u proxy_eval.py $tag cuda $data 2>&1 | grep proxy | tee -a "$DRV/logs/day3.log"
    cp "$OUT/$tag/iso_val.npz" "$OUT/$tag/proxy_logits.npz" "$DRV/outputs/$tag/" 2>/dev/null ) &
  sleep 150; }

# the 50 Hz run can start right away; the 100 Hz data is built meanwhile
run r5_ep60 50 subj0_pca_w3 --w3aug --layers 8 --epochs 60 --ema 0.999
python -u download_subj0.py > "$DRV/logs/day3_download.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py runs 10 > "$DRV/logs/day3_w3hprep.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py holdout >> "$DRV/logs/day3_w3hprep.log" 2>&1
rm -rf /content/pnpl/data/raw/LibriBrain /content/pnpl/data/raw/LibriBrain2
log "w3h data: $(ls $DER/subj0_pca_w3h/runs | wc -l) runs; $(tail -1 "$DRV/logs/day3_w3hprep.log")"
run r5_h100 100 subj0_pca_w3h --w3aug --layers 8 --epochs 40 --ema 0.999
wait
log "DAY3_DONE"
