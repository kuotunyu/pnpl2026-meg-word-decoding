#!/bin/bash
# Local RTX 4090 10/13 (~4.5 h): is the wide CNN also better for 1 s windows? (day15: 3 s CNN width 512 = best model)
#  rC1_s1: validation-split 1 s, CNN width 512   rE1_s1: same setting, default width 320 (control)
#  -> proxy + iso validation for both; then feC_s7: full-data 1 s CNN width 512 (replacement candidate for a 1 s member)
cd "$(dirname "$0")"
P=${PY:-python}
LOG=../outputs/local_queue.log
log() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $LOG; }
val() { tag=$1; shift
  $P -u train_ctx.py --tag $tag --layers 8 --epochs 40 --ema 0.999 "$@" > ../outputs/$tag.log 2>&1
  log "$tag: $(grep -h -E 'DONE|Error' ../outputs/$tag.log | tail -1 | cut -c1-170)"
  $P -u iso_eval.py $tag cuda 2>&1 | grep iso | tee -a $LOG
  $P -u proxy_eval.py $tag cuda 2>&1 | grep proxy | tee -a $LOG; }
log "queue4 start"
val rC1_s1 --seed 41 --width 512
val rE1_s1 --seed 41
$P -u train_ctx.py --tag feC_s7 --full --seed 47 --layers 8 --epochs 40 --ema 0.999 --width 512 > ../outputs/feC_s7.log 2>&1
log "feC_s7: $(grep -h -E 'DONE|Error' ../outputs/feC_s7.log | tail -1 | cut -c1-150)"
log "QUEUE4_DONE"
