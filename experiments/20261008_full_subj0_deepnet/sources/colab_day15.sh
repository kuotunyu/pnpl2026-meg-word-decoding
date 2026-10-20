#!/bin/bash
# Colab night 10/12-13 (~4.5 h): VALIDATION of the two levers Public can no longer rank (Public too sensitive, 10/12).
# Validation-split (not --full) 3 s models, scored on the holdout-like proxy (1042 words) + isolated-word validation:
#  slot A: rL_s1   12-layer d768            (architecture diversity, like w3L)
#  slot B: rC_s1   CNN width 512            (architecture diversity, like w3C)
#  slot C: rp4_ema self-training of r4_ema with teacher B5 (does holdout self-training hurt/help generalisation?)
#          then r4b_s2 = r4_ema setting, seed 2 (control: "one more same-architecture member")
# Afterwards (locally): proxy ensembles base{r1_w3,r4_ema,r5_ep60} + {rL,rC} vs + {r4b} vs + {rp4}.
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day15.log"; }
log "start"
python -u safe_copy.py "$DRV/derived_w3/subj0_pca_w3" "$DER/subj0_pca_w3" 2>&1 | tail -1 | tee -a "$DRV/logs/day15.log"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
mkdir -p "$OUT/r4_ema"; cp "$DRV/outputs/r4_ema/model.pt" "$OUT/r4_ema/"
ls -l "$OUT/r4_ema/model.pt" "$DRV/outputs/r4_ema/model.pt" | tee -a "$DRV/logs/day15.log"
evals() { tag=$1
  python -u iso_eval.py $tag cuda subj0_pca_w3 2>&1 | grep iso | tee -a "$DRV/logs/day15.log"
  python -u proxy_eval.py $tag cuda subj0_pca_w3 2>&1 | grep proxy | tee -a "$DRV/logs/day15.log"
  mkdir -p "$DRV/outputs/$tag"; cp "$OUT/$tag/iso_val.npz" "$OUT/$tag/proxy_logits.npz" "$DRV/outputs/$tag/" 2>/dev/null; }
train() { tag=$1; shift
  python -u train_ctx.py --tag $tag --data subj0_pca_w3 --w3aug --epochs 40 --ema 0.999 "$@" > "$DRV/logs/$tag.log" 2>&1
  cp -r "$OUT/$tag" "$DRV/outputs/"
  log "$tag finished: $(grep -h -E 'DONE|Error' "$DRV/logs/$tag.log" | tail -1 | cut -c1-170)"
  evals $tag; }
( train rL_s1 --seed 31 --layers 12 --d 768 ) &
sleep 150
( train rC_s1 --seed 32 --layers 8 --width 512 ) &
sleep 150
( python -u pseudo_ft.py --tag rp4_ema --init r4_ema --teacher teacher_B5.csv --data subj0_pca_w3 --w3aug --seed 141 > "$DRV/logs/rp4_ema.log" 2>&1
  log "rp4_ema: $(grep -h -E 'DONE|Error' "$DRV/logs/rp4_ema.log" | tail -1 | cut -c1-150)"
  evals rp4_ema
  train r4b_s2 --seed 2 --layers 8 ) &
wait
log "DAY15_DONE"
