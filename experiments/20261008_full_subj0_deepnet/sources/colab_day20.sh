#!/bin/bash
# Colab 10/13 evening (~1.5 h): third self-training round on the PROVEN model family with teacher final_B5i.
# Public history of holdout self-training: W 0.679 -> P1 (teacher W) 0.694 -> B2 (P1-teacher members) 0.698 -> B5 0.710.
#  slot A: vw3_s1-3 (init w3f_s1-3)   slot B: vw3_s4-6 (init w3f_s4-6)   slot C: vf7_s1-3 (init f7_s1-3, 1 s)
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day20.log"; }
log "start"
# verified copies (10/12 first try: plain cp -r gave truncated .npz files -> every job failed with EOFError)
python -u safe_copy.py "$DRV/derived_w3/subj0_pca_w3" "$DER/subj0_pca_w3" 2>&1 | tail -1 | tee -a "$DRV/logs/day20.log"
python -u safe_copy.py "$DRV/derived/subj0_pca_epochs" "$DER/subj0_pca_epochs" 2>&1 | tail -1 | tee -a "$DRV/logs/day20.log"
df -h /content | tail -1 | tee -a "$DRV/logs/day20.log"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
[ -f "$DER/subj0_pca_epochs/holdout_subj00.npz" ] || python -u preprocess.py holdout 2>&1 | tail -1 | tee -a "$DRV/logs/day20.log"
for t in w3f_s1 w3f_s2 w3f_s3 w3f_s4 w3f_s5 w3f_s6 f7_s1 f7_s2 f7_s3; do mkdir -p "$OUT/$t"; cp "$DRV/outputs/$t/model.pt" "$OUT/$t/"; ls -l "$OUT/$t/model.pt" "$DRV/outputs/$t/model.pt" | tee -a "$DRV/logs/day20.log"; done
train() { tag=$1; data=$2; shift 2
  python -u train_ctx.py --tag $tag --data $data --full --epochs 40 --ema 0.999 "$@" > "$DRV/logs/$tag.log" 2>&1
  cp -r "$OUT/$tag" "$DRV/outputs/"
  log "$tag finished: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-120) holdout=$(ls "$OUT/$tag/holdout_logits.npz" 2>/dev/null | wc -l)"; }
ft() { tag=$1; shift
  python -u pseudo_ft.py --tag $tag --teacher teacher_B5i.csv "$@" > "$DRV/logs/$tag.log" 2>&1
  mkdir -p "$DRV/outputs/$tag"; cp "$OUT/$tag/holdout_logits.npz" "$OUT/$tag/model.pt" "$DRV/outputs/$tag/" 2>/dev/null
  log "$tag: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-150)"; }
( ft vw3_s1 --init w3f_s1 --data subj0_pca_w3 --w3aug --seed 201
  ft vw3_s2 --init w3f_s2 --data subj0_pca_w3 --w3aug --seed 202
  ft vw3_s3 --init w3f_s3 --data subj0_pca_w3 --w3aug --seed 203 ) &
sleep 60
( ft vw3_s4 --init w3f_s4 --data subj0_pca_w3 --w3aug --seed 204
  ft vw3_s5 --init w3f_s5 --data subj0_pca_w3 --w3aug --seed 205
  ft vw3_s6 --init w3f_s6 --data subj0_pca_w3 --w3aug --seed 206 ) &
sleep 60
( ft vf7_s1 --init f7_s1 --data subj0_pca_epochs --seed 211
  ft vf7_s2 --init f7_s2 --data subj0_pca_epochs --seed 212
  ft vf7_s3 --init f7_s3 --data subj0_pca_epochs --seed 213 ) &
wait
log "DAY20_DONE"
