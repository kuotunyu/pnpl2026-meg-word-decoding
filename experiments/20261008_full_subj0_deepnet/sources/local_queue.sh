#!/bin/bash
# Local RTX 4090 overnight queue 10/12 (one GPU job at a time). Local-PCA 1 s data (data/derived/subj0_pca_epochs).
#  1-3: self-training (teacher final_P1) of the local full-data 1 s models f_s1..s3 -> pfl_s1..s3
#  4-5: two new local full-data 1 s models, c7 setting + EMA -> fe_s4, fe_s5
#  6-7: self-training of fe_s4, fe_s5 -> pfe_s4, pfe_s5
cd "$(dirname "$0")"
P=${PY:-python}
LOG=../outputs/local_queue.log
log() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $LOG; }
log "start"
ft() { tag=$1; shift; $P -u pseudo_ft.py --tag $tag --teacher teacher_P1.csv --data subj0_pca_epochs "$@" > ../outputs/$tag.log 2>&1; log "$tag: $(grep -h -E 'DONE|Error' ../outputs/$tag.log | tail -1 | cut -c1-150)"; }
tr() { tag=$1; shift; $P -u train_ctx.py --tag $tag --full "$@" > ../outputs/$tag.log 2>&1; log "$tag: $(grep -h -E 'DONE|Error' ../outputs/$tag.log | tail -1 | cut -c1-150)"; }
ft pfl_s1 --init f_s1 --seed 31
ft pfl_s2 --init f_s2 --seed 32
ft pfl_s3 --init f_s3 --seed 33
tr fe_s4 --seed 44 --layers 8 --epochs 40 --ema 0.999
tr fe_s5 --seed 45 --layers 8 --epochs 40 --ema 0.999
ft pfe_s4 --init fe_s4 --seed 34
ft pfe_s5 --init fe_s5 --seed 35
log "QUEUE_DONE"
