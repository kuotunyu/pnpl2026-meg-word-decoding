#!/bin/bash
# Build one submission CSV on the local CPU from models whose holdout_logits.npz are in outputs/<tag>/.
#   [LMFLAGS=--eos] bash sources/build_final.sh <name> "<sentence-row tags>" "<isolated-row tags>"
# Sentence rows: mean log-probs of the tags (repeat a tag to up-weight it) fused with GPT-2 and gpt2-medium
# closed-vocab soft PMI, strength-calibrated average, alpha 0.5 (lm_fusion.py + lm_calib.py).
# Isolated rows: mean of context and context-free heads of the iso tags, tau 0.7 (make_final.py).
# Output: outputs/final_<name>.csv (+ fused_<name>_* intermediates). Run from the stage folder.
set -e
NAME=$1; TAGS=$2; ISO=$3
P=${PY:-python}
O=outputs
for lm in gpt2 gpt2-medium; do
  $P -u sources/lm_fusion.py --tags $TAGS --lm $lm --alpha 0.5 --beam 32 --soft --pmi --device ${LMDEV:-cpu} $LMFLAGS \
     --out $O/fused_${NAME}_$lm.csv > $O/fused_${NAME}_$lm.log 2>&1
done
$P sources/lm_calib.py --tags $TAGS --mix $O/fused_${NAME}_gpt2_lmmix.npy $O/fused_${NAME}_gpt2-medium_lmmix.npy \
   --alpha 0.5 --out $O/fused_${NAME}_calib2.csv
$P sources/make_final.py --fused $O/fused_${NAME}_calib2.csv --tags $TAGS --iso_tags $ISO --tau 0.7 --out $O/final_$NAME.csv
echo "built $O/final_$NAME.csv"
