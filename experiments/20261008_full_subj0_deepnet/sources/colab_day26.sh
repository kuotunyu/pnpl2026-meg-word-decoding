#!/bin/bash
# Colab 10/14 (~0.5 h part 1, +1.5 h part 2): TEST-TIME INPUT-SCALE FIX for every model of B5i.
# Holdout windows are ~1.35x larger per PCA component than training windows (whole-run vs own-segment z-score);
# infer_rescale.py rescales each component to the training std and re-runs inference -> holdout_logits_rs.npz.
SRC=/content/pnpl/experiments/20261008_full_subj0_deepnet/sources
DRV="/content/drive/MyDrive/Kaggle專案/PNPL_Deep_2026/experiments/20261008_full_subj0_deepnet"
OUT=/content/pnpl/experiments/20261008_full_subj0_deepnet/outputs
DER=/content/pnpl/data/derived
mkdir -p "$DRV/outputs" "$DRV/logs" "$OUT" "$DER"
cd "$SRC"
log() { echo "$(date -u +%H:%M)UTC $*" | tee -a "$DRV/logs/day26.log"; }
log "start"
# verified copies (10/12 first try: plain cp -r gave truncated .npz files -> every job failed with EOFError)
python -u safe_copy.py "$DRV/derived_w3/subj0_pca_w3" "$DER/subj0_pca_w3" 2>&1 | tail -1 | tee -a "$DRV/logs/day26.log"
python -u safe_copy.py "$DRV/derived/subj0_pca_epochs" "$DER/subj0_pca_epochs" 2>&1 | tail -1 | tee -a "$DRV/logs/day26.log"
df -h /content | tail -1 | tee -a "$DRV/logs/day26.log"
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["COMPETITION_HOLDOUT/subj00_holdout01_sentence.npz", "COMPETITION_HOLDOUT/subj00_holdout2_word.npz"]:
    hf_hub_download("pnpl/LibriBrain-Competition-2026", f, repo_type="dataset", local_dir="/content/pnpl/data/raw/LibriBrain-Competition-2026")
PY
[ -f "$DER/subj0_pca_epochs/holdout_subj00.npz" ] || python -u preprocess.py holdout 2>&1 | tail -1 | tee -a "$DRV/logs/day26.log"
getm() { for t in "$@"; do mkdir -p "$OUT/$t"; cp "$DRV/outputs/$t/model.pt" "$DRV/outputs/$t/holdout_logits.npz" "$OUT/$t/" 2>/dev/null; done; }
putrs() { for t in "$@"; do [ -f "$OUT/$t/holdout_logits_rs.npz" ] && cp "$OUT/$t/holdout_logits_rs.npz" "$DRV/outputs/$t/"; done; log "saved rs: $(for t in "$@"; do [ -f "$OUT/$t/holdout_logits_rs.npz" ] && echo -n "$t "; done)"; }
W3="pw3_s1 pw3_s2 pw3_s3 w3f_s1 w3f_s2 w3f_s3 w3f_s4 w3f_s5 w3f_s6 w3f_s7 w3f_s8 rw3_s7 rw3_s8 rw3e_s12 rw3d_s13 w3e_s14 w3e_s15 w3ef_s12 w3df_s13 w3L_s21 w3C_s22"
EP="pf7_s1 f7_s1 f7_s2 f7_s3 rf7e_s4 rf7e_s5 f7e_s4 f7e_s5 f7e_s6"
H="rw3h_s1 rw3h_s2 rw3h_s3 w3hf_s1 w3hf_s2 w3hf_s3 w3hf_s4"
getm $W3 $EP
python -u infer_rescale.py --data subj0_pca_w3 --tags $W3 2>&1 | grep -v -i warn | grep -v TransformerEncoder | tee -a "$DRV/logs/day26.log"
putrs $W3
python -u infer_rescale.py --data subj0_pca_epochs --tags $EP 2>&1 | grep -v -i warn | grep -v TransformerEncoder | tee -a "$DRV/logs/day26.log"
putrs $EP
log "PART1_DONE (3 s 50 Hz + 1 s Colab PCA)"
# part 2: 100 Hz models need the 100 Hz windows rebuilt from Hugging Face (~1-1.5 h)
rm -rf "$DER/subj0_pca_w3"
python -u download_subj0.py > "$DRV/logs/day26_download.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py runs 10 > "$DRV/logs/day26_w3hprep.log" 2>&1
PNPL_FS=100 python -u preprocess_w3.py holdout >> "$DRV/logs/day26_w3hprep.log" 2>&1
rm -rf /content/pnpl/data/raw/LibriBrain /content/pnpl/data/raw/LibriBrain2
log "w3h data: $(ls $DER/subj0_pca_w3h/runs | wc -l) runs"
getm $H
PNPL_FS=100 python -u infer_rescale.py --data subj0_pca_w3h --tags $H 2>&1 | grep -v -i warn | grep -v TransformerEncoder | tee -a "$DRV/logs/day26.log"
putrs $H
log "DAY26_DONE"
