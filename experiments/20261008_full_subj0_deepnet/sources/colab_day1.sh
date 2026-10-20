#!/bin/bash
# Colab day run 10/09 (~8-10 h, unattended). All results + logs go to Drive as soon as each piece finishes.
#  Phase A (validation + specialist): r2_w4 (4 s window), r2_iso (isolated-word specialist, val), isof_s1 (specialist, full data)
#  Phase B: the better window (w4 if it beats w3 by > 0.005 novel_ctx) gets 2 more full-data seeds
#  Phase C: final ensembles with LM fusion (GPT-2 / Qwen3-14B / Qwen3-32B / GPT-2+Qwen average), two isolated-row variants
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
mkdir -p "$DRV/outputs" "$DRV/logs" "$DRV/submissions" "$OUT"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day1.log"; }
log "start"

# ---- data
mkdir -p /content/pnpl/data/derived
mkdir -p /content/pnpl/data/derived/subj0_pca_epochs && cp "$DRV/derived/subj0_pca_epochs/pca_W.npy" /content/pnpl/data/derived/subj0_pca_epochs/  # same PCA basis as all Colab models
[ -d /content/pnpl/data/derived/subj0_pca_w3/runs ] || cp -r "$DRV/derived_w3/subj0_pca_w3" /content/pnpl/data/derived/
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
pip -q install hf_xet 2>&1 | tail -1
python -u download_subj0.py > "$DRV/logs/day1_download.log" 2>&1
PNPL_POST=3.5 python -u preprocess_w3.py runs 10 > "$DRV/logs/day1_w4prep.log" 2>&1
PNPL_POST=3.5 python -u preprocess_w3.py holdout >> "$DRV/logs/day1_w4prep.log" 2>&1
rm -rf /content/pnpl/data/raw/LibriBrain /content/pnpl/data/raw/LibriBrain2   # free ~96 GB for Qwen3-32B; holdout + header file kept
df -h /content | tail -1 | tee -a "$DRV/logs/day1.log"
log "data ready: w3 $(ls /content/pnpl/data/derived/subj0_pca_w3/runs | wc -l) runs, w4 $(ls /content/pnpl/data/derived/subj0_pca_w4/runs | wc -l) runs"
for t in f7_s1 f7_s2 f7_s3 r1_w3 w3f_s1 w3f_s2 w3f_s3 w3f_s4 w3f_s5 w3f_s6; do
  [ -d "$OUT/$t" ] || cp -r "$DRV/outputs/$t" "$OUT/" 2>/dev/null || log "missing on Drive: $t"
done

run() { tag=$1; shift
  ( python -u train_ctx.py --tag $tag "$@" > "$DRV/logs/$tag.log" 2>&1
    cp -r "$OUT/$tag" "$DRV/outputs/"
    log "$tag finished: $(grep -h DONE "$DRV/logs/$tag.log" | cut -c1-170)" ) &
  sleep 150; }

# ---- Phase A
log "phase A"
run r2_w4  --data subj0_pca_w4 --w3aug --layers 8 --epochs 40
run r2_iso --data subj0_pca_w3 --iso --bs 1024 --layers 8 --epochs 30
run isof_s1 --full --seed 1 --data subj0_pca_w3 --iso --bs 1024 --layers 8 --epochs 30
wait
python -u iso_eval.py r1_w3 cuda subj0_pca_w3 2>&1 | grep iso | tee -a "$DRV/logs/day1.log"
python -u iso_eval.py r2_iso cuda subj0_pca_w3 2>&1 | grep iso | tee -a "$DRV/logs/day1.log"

# ---- Phase B
W4=$(python -c "import json;print(json.load(open('$OUT/r2_w4/metrics.json'))['novel_ctx'])" 2>/dev/null || echo 0)
W3=$(python -c "import json;print(json.load(open('$OUT/r1_w3/metrics.json'))['novel_ctx'])" 2>/dev/null || echo 0.6505)
if python -c "import sys; sys.exit(0 if $W4 > $W3 + 0.005 else 1)"; then
  log "phase B: 4 s window wins (w4 $W4 vs w3 $W3) -> w4f_s1, w4f_s2"
  run w4f_s1 --full --seed 1 --data subj0_pca_w4 --w3aug --layers 8 --epochs 40
  run w4f_s2 --full --seed 2 --data subj0_pca_w4 --w3aug --layers 8 --epochs 40
else
  log "phase B: keep 3 s window (w4 $W4 vs w3 $W3) -> w3f_s7, w3f_s8"
  run w3f_s7 --full --seed 7 --data subj0_pca_w3 --w3aug --layers 8 --epochs 40
  run w3f_s8 --full --seed 8 --data subj0_pca_w3 --w3aug --layers 8 --epochs 40
fi
wait

# ---- Phase C: final candidates
LONG=$(cd "$OUT" && ls -d w3f_s* w4f_s* 2>/dev/null | while read t; do [ -f "$t/holdout_logits.npz" ] && echo -n "$t "; done)
ENS="$LONG f7_s1 f7_s2 f7_s3"
log "phase C ensemble: $ENS"
for spec in "gpt2:gpt2" "qwen3_14b:Qwen/Qwen3-14B" "qwen3_32b:Qwen/Qwen3-32B"; do
  name=${spec%%:*}; lm=${spec#*:}
  python -u lm_fusion.py --tags $ENS --lm $lm --alpha 0.5 --beam 32 --soft --pmi --device cuda --out "$OUT/fusedE_$name.csv" > "$DRV/logs/day1_lm_$name.log" 2>&1
  log "fused $name: $(tail -1 "$DRV/logs/day1_lm_$name.log" | cut -c1-120)"
done
python lm_avg.py --inputs "$OUT/fusedE_gpt2.csv" "$OUT/fusedE_qwen3_14b.csv" "$OUT/fusedE_qwen3_32b.csv" --out "$OUT/fusedE_lmavg.csv" | tee -a "$DRV/logs/day1.log"
ISO_SPEC=$( [ -f "$OUT/isof_s1/holdout_logits.npz" ] && echo "isof_s1 $LONG" || echo "$LONG" )
for name in gpt2 qwen3_14b qwen3_32b lmavg; do
  [ -f "$OUT/fusedE_$name.csv" ] || continue
  python make_final.py --fused "$OUT/fusedE_$name.csv" --tags $ENS --iso_tags $LONG --tau 0.7 --out "$OUT/finalE_$name.csv"
  python make_final.py --fused "$OUT/fusedE_$name.csv" --tags $ENS --iso_tags $ISO_SPEC --tau 0.7 --out "$OUT/finalE_${name}_isospec.csv"
  cp "$OUT/finalE_$name.csv" "$OUT/finalE_${name}_isospec.csv" "$DRV/submissions/"
done
log "DAY1_DONE submissions: $(ls "$DRV/submissions" | tr '\n' ' ')"
