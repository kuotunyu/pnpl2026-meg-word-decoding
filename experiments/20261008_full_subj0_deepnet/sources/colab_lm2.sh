#!/bin/bash
# Second Colab runtime, 10/11 (~1-1.5 h, GPU only for LM inference): LM comparison WITH strength calibration.
# For each LM: (1) holdout soft-PMI mixture for the accepted MEG ensemble W (lm_fusion.py, saves *_lmmix.npy)
#              (2) the same mixture on the holdout-like validation segments (proxy_mix.py)
# Everything goes to Drive  .../20261008_full_subj0_deepnet/lm2/ ; the runtime releases itself at the end.
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
mkdir -p "$DRV/lm2" "$DRV/logs" "$OUT"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/lm2.log"; }
log "start"
W="w3f_s1 w3f_s2 w3f_s3 w3f_s4 w3f_s5 w3f_s6 f7_s1 f7_s2 f7_s3"
PX="r1_w3 r4_ema r5_ep60 r1_c7ref"
for t in $W; do mkdir -p "$OUT/$t"; cp "$DRV/outputs/$t/holdout_logits.npz" "$OUT/$t/"; done
for t in $PX; do mkdir -p "$OUT/$t"; cp "$DRV/outputs/$t/proxy_logits.npz" "$OUT/$t/"; done
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
pip -q install 'transformers==4.56.2' 2>&1 | tail -1
for spec in "gpt2:gpt2" "gpt2med:gpt2-medium" "gpt2large:gpt2-large" "gpt2xl:gpt2-xl" "qwen3_1p7b:Qwen/Qwen3-1.7B" "qwen3_8b:Qwen/Qwen3-8B" "qwen3_14b:Qwen/Qwen3-14B" "llama32_3b:unsloth/Llama-3.2-3B"; do
  name=${spec%%:*}; lm=${spec#*:}
  python -u lm_fusion.py --tags $W --lm $lm --alpha 0.5 --beam 32 --soft --pmi --device cuda --out "$OUT/fusedL_$name.csv" > "$DRV/logs/lm2_$name.log" 2>&1
  python -u proxy_mix.py --tags $PX --lm $lm --name $name --device cuda >> "$DRV/logs/lm2_$name.log" 2>&1
  cp "$OUT/fusedL_${name}_lmmix.npy" "$OUT/proxy_mix_$name.npy" "$DRV/lm2/" 2>/dev/null
  log "$name: $(ls "$DRV/lm2/" | grep -c "$name") files; $(grep -h -E 'Error|wrote' "$DRV/logs/lm2_$name.log" | tail -2 | tr '\n' ' ' | cut -c1-200)"
done
log "LM2_DONE"
