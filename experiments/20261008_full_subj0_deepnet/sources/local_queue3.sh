#!/bin/bash
# Local RTX 4090 10/12-13 (~2.5 h): self-training with teacher B5 on local 1 s models (ufl/ufe, ~10 min each),
# then an architecture-diverse local 1 s model (12-layer Transformer, d 768, EMA) -> feL_s6 (~2 h).
cd "$(dirname "$0")"
P=${PY:-python}
LOG=../outputs/local_queue.log
log() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $LOG; }
ft() { tag=$1; shift; $P -u pseudo_ft.py --tag $tag --teacher teacher_B5.csv --data subj0_pca_epochs "$@" > ../outputs/$tag.log 2>&1; log "$tag: $(grep -h -E 'DONE|Error' ../outputs/$tag.log | tail -1 | cut -c1-150)"; }
log "queue3 start"
ft ufl_s1 --init f_s1 --seed 101
ft ufl_s2 --init f_s2 --seed 102
ft ufe_s4 --init fe_s4 --seed 104
ft ufe_s5 --init fe_s5 --seed 105
$P -u train_ctx.py --tag feL_s6 --full --seed 46 --layers 12 --d 768 --epochs 40 --ema 0.999 > ../outputs/feL_s6.log 2>&1
log "feL_s6: $(grep -h -E 'DONE|Error' ../outputs/feL_s6.log | tail -1 | cut -c1-150)"
log "QUEUE3_DONE"
