#!/bin/bash
# Local RTX 4090 queue 10/12: self-training with teacher final_B2 of the local-PCA 1 s models -> tfl_s1..3, tfe_s4, tfe_s5
cd "$(dirname "$0")"
P=${PY:-python}
LOG=../outputs/local_queue.log
log() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $LOG; }
log "queue2 start (teacher B2)"
ft() { tag=$1; shift; $P -u pseudo_ft.py --tag $tag --teacher teacher_B2.csv --data subj0_pca_epochs "$@" > ../outputs/$tag.log 2>&1; log "$tag: $(grep -h -E 'DONE|Error' ../outputs/$tag.log | tail -1 | cut -c1-150)"; }
ft tfl_s1 --init f_s1 --seed 51
ft tfl_s2 --init f_s2 --seed 52
ft tfl_s3 --init f_s3 --seed 53
ft tfe_s4 --init fe_s4 --seed 54
ft tfe_s5 --init fe_s5 --seed 55
log "QUEUE2_DONE"
