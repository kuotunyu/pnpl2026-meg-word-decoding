#!/bin/bash
# Colab: copy derived data from Drive to local disk, then train full-data seeds of one config in parallel
# (A100-80GB holds three 16 GB copies). Usage: bash colab_train.sh <tag_prefix> "<train_ctx args>" <seed...>
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
PREFIX=$1; ARGS=$2; shift 2
mkdir -p /content/pnpl/data/derived "$DRV/outputs" "$DRV/logs"
if [ ! -d /content/pnpl/data/derived/subj0_pca_epochs/runs ]; then cp -r "$DRV/derived/"* /content/pnpl/data/derived/; fi
echo "runs: $(ls /content/pnpl/data/derived/subj0_pca_epochs/runs | wc -l)"
cd "$SRC"
for s in "$@"; do
  ( python -u train_ctx.py --tag ${PREFIX}_s$s --full --seed $s $ARGS > "$DRV/logs/${PREFIX}_s$s.log" 2>&1
    cp -r ../outputs/${PREFIX}_s$s "$DRV/outputs/" && echo "COPIED ${PREFIX}_s$s" >> "$DRV/logs/${PREFIX}_s$s.log" ) &
  sleep 90   # stagger data loading
done
wait
echo ALL_DONE
