#!/bin/bash
# Reproduce the post-first-review pipeline end to end (≈ 1.5 h on an M-series laptop; models download on first run).
set -e; cd "$(dirname "$0")"
python3 ../audit/00_metadata.py
python3 01_prepare.py            # normalised 1.5 s windows
python3 02_handfeat.py           # 54 hand-crafted descriptors
python3 04_audit.py              # Checks A-E + sample-rate leak test
python3 06_checkF.py             # Check F: non-speech control
for m in microsoft/wavlm-base-plus facebook/wav2vec2-large-xlsr-53; do python3 03_ssl_embed.py $m norm; done
python3 03b_ecapa.py             # ECAPA-TDNN (SpeechBrain)
python3 05_evaluate.py           # LOSO / TEST / cross-lingual
python3 08_robustness.py && python3 02b_handfeat_chanrand.py && python3 09_robust_eval.py
python3 10_final_model.py && python3 07_report.py
