#!/bin/bash
# Colab night run 10/08: (1) LM comparison on the same MEG ensemble, (2) round-1 architecture/window experiments.
# Expects step1 (code bundle incl. local holdout logits) and colab_train.sh (derived data in /content) done.
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
mkdir -p "$DRV/outputs" "$DRV/logs" "$DRV/submissions"
cd "$SRC"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
# local models' holdout logits shipped in the bundle + Colab f7 seeds
for t in f7_s1 f7_s2 f7_s3; do [ -d ../outputs/$t ] || cp -r "$DRV/outputs/$t" ../outputs/; done
TAGS="f7_s1 f7_s2 f7_s3 f_s1 f_s2 f_s3 c5_dedup c2_ep30 c7_l8"
pip -q install -U transformers accelerate 2>&1 | tail -1
for spec in "gpt2:gpt2" "qwen3_8b:Qwen/Qwen3-8B" "qwen3_14b:Qwen/Qwen3-14B"; do
  name=${spec%%:*}; lm=${spec#*:}
  python -u lm_fusion.py --tags $TAGS --lm $lm --alpha 0.5 --beam 32 --soft --pmi --device cuda --out ../outputs/fused_$name.csv > "$DRV/logs/lm_$name.log" 2>&1
  python make_final.py --fused ../outputs/fused_$name.csv --tags $TAGS --tau 0.7 --out ../outputs/final_$name.csv >> "$DRV/logs/lm_$name.log" 2>&1
  cp ../outputs/final_$name.csv "$DRV/submissions/" && echo "LM $name done"
done
# round 1: long windows + deeper/wider transformer, all with validation (compare novel_ctx)
[ -d /content/pnpl/data/raw/LibriBrain2 ] || python -u download_subj0.py > "$DRV/logs/download.log" 2>&1
python -u preprocess_w3.py runs 10 > "$DRV/logs/w3_prep.log" 2>&1
python -u preprocess_w3.py holdout >> "$DRV/logs/w3_prep.log" 2>&1
echo "w3 runs: $(ls /content/pnpl/data/derived/subj0_pca_w3/runs | wc -l)"
run() { tag=$1; shift; ( python -u train_ctx.py --tag $tag "$@" > "$DRV/logs/$tag.log" 2>&1; cp -r ../outputs/$tag "$DRV/outputs/"; echo "$tag finished" ) & sleep 120; }
run r1_c7ref --layers 8 --epochs 40
run r1_l12d768 --layers 12 --d 768 --epochs 40
run r1_w3 --data subj0_pca_w3 --w3aug --layers 8 --epochs 40
wait
grep -h DONE "$DRV"/logs/r1_*.log
echo NIGHT1_DONE
