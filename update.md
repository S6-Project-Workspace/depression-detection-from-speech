# Team 10 · Speech Processing · Project Update (First Review → Final Review)

*Acoustic depression screening from Tamil & Malayalam speech.*
Everything below is reproducible from `final_pipeline/` (`run_all.sh`); all numbers come from the JSON files in `final_pipeline/results/`.

---

## 1. TL;DR

* The **whole pipeline of the first-review block diagram is now implemented end to end** — corpus audit → channel/duration normalisation → hand-crafted features → ECAPA/WavLM/XLSR-53 embeddings → source-aware evaluation → cross-lingual test → robustness study → a saved final model with a `predict.py` CLI that returns *risk + confidence* (block 7 of the schematic).
* The first review warned that the 0.89–0.91 baselines were inflated by a recording-protocol confound. **That warning was right, and the remediation work shows it is worse than we thought:** the confound is not just sample rate / duration — it survives band-limiting, fixed windows, noise-floor equalisation, source-held-out validation, and even a non-speech-only control (Section 4).
* Consequence: on the organiser's test set **every representation (hand-crafted, ECAPA-style, WavLM, XLSR-53) scores ≈ 1.00 AUC**, including Tamil→Malayalam transfer. The metric is saturated and cannot rank models, so "improve accuracy" is no longer a meaningful goal on its own.
* We therefore reframed the remaining 5 days around **what *can* still be measured and improved: robustness to the recording channel** (channel-randomisation training lifts balanced accuracy under a random channel from 0.58 → 0.96 for hand-crafted features and 0.88 → 0.99 for WavLM, Section 5) plus fine-tuning / channel-invariant training (Section 8).

---

## 2. Status against the first-review work plan (Slide 14)

| # | First-review task | Status | Evidence / where |
|---|---|---|---|
| 1 | Per-session / per-device normalisation + domain adaptation | **Done** (3 variants evaluated) | `common.py` (`normalise`, `noise_floor_equalise`, `normalise_v2`), `05_evaluate.py` nuisance-subspace projection |
| 2 | Re-run audit Checks A–E; confirm near-chance metadata performance | **Done — result negative** (confound persists) | `04_audit.py`, `06_checkF.py` (new Check F), `results/audit.json`, `results/checkF.json` |
| 3 | Speaker-disjoint, condition-matched splits | **Partly possible** — *source-disjoint* (leave-one-recording-source-out) done; true speaker-disjoint and condition-matched are **not achievable from this corpus** (Section 6) | `05_evaluate.py` (`LOSO_pairs`) |
| 4 | ECAPA-TDNN 192-d embeddings + MLP/linear head | **Done** (see note on ECAPA in §4.5) | `03b_ecapa.py`, `ecapa_model.py` (legacy), `05_evaluate.py` |
| 5 | wav2vec 2.0 / WavLM on the audited corpus | **Done as frozen layer-wise probes** (full fine-tuning deliberately deferred — see §8) | `03_ssl_embed.py`, layer scan `results/fig_layerscan.png` |
| 6 | Cross-lingual: train Tamil, test Malayalam (and reverse) | **Done** | `results/eval_results.json` (`XLING_*`), `results/fig_robustness.png` |
| 7 | Feature ablation study + final report prep | **Done** (ablation, Check F, robustness); report slides still to be assembled | `04_audit.py` Check C, this document |

---

## 3. What was built (all in `final_pipeline/`)

| Stage | Script | Output |
|---|---|---|
| 0 | `../audit/00_metadata.py` | container metadata + MD5 for all 3,422 clips |
| 1 | `01_prepare.py` | 16 kHz mono, 80 Hz–3.8 kHz band-limited, RMS-normalised, silence-trimmed **fixed 1.5 s windows (≤ 3 per clip)** → 8,730 windows |
| 2 | `02_handfeat.py` | 54 descriptors / window: F0 (3), ZCR/RMS (4), centroid/bandwidth/roll-off/flatness (8), 13 MFCC mean+std + Δ (39) |
| 3 | `03_ssl_embed.py`, `03b_ecapa.py` | layer-wise mean-pooled embeddings: WavLM-base-plus (13 layers), wav2vec 2.0 XLSR-53 (25 layers), ECAPA-TDNN (192-d) |
| 4 | `04_audit.py`, `06_checkF.py` | Checks A–E (re-run) + **Check F (new)** + sample-rate leak test |
| 5 | `05_evaluate.py` | LOSO / organiser-TEST / cross-lingual, LR & SVM heads, optional nuisance-subspace projection |
| 8–9 | `08_robustness.py`, `09_robust_eval.py` | random-recording-channel test set + channel-augmented training |
| 10 | `10_final_model.py`, `predict.py` | saved model `model/wavlm_l6_lr.joblib`; CLI scorer |
| 7 | `07_report.py` | figures + `results/results_table.md` |

Design decisions that follow from the first review:
* **Fixed-length windows** (instead of whole clips) remove the duration cue (Tamil: depressed 6.5 s vs non-depressed 2.5 s → duration alone is 98 % predictive).
* **Same band-limit for every clip** removes the bandwidth part of the sample-rate cue.
* **Clip-level decision = mean of window scores**; macro-F1 / balanced accuracy / AUC reported (Tamil train is 1:2 imbalanced).

---

## 4. Findings (this is the story for the final review)

### 4.1 The confound, fully characterised

Recording sources recovered from the file names (`D_A*`, `D_F*`, `D_S*` = depressed; `ND1`–`ND5` = non-depressed batches):

| Language | Depressed sources (native sr) | Non-depressed sources (native sr) |
|---|---|---|
| Tamil | A (16 k, 80), F (16 k, 133), S (16 k, 241) — **all 16 kHz** | ND1–5 (**all 48 kHz**, 920 clips) |
| Malayalam | A (16 k, 248), S (16 k, 256), F (**48 k**, 284) | ND1–5 (**all 48 kHz**, 900 clips) |

Organiser test sets have the same structure (Tamil: all D @16 k, all ND @48 k; Malayalam: D @16 k and @48 k, ND @48 k only).

### 4.2 Audit Checks A–E re-run (`results/audit.json`)

| Check | Result |
|---|---|
| A  descriptives | Mean clip duration D vs ND: Tamil 6.5 s vs 2.5 s; Malayalam 5.3 s vs 4.2 s. ZCR differs 1.4× (Tamil) / 2.3× (Malayalam) on resample-only audio. |
| B  metadata-only | **Tamil: native sr alone = 100 % CV / 100 % test accuracy**; duration alone = 98 % / 97 %. Malayalam: sr alone 83 % / 82 %, sr+duration 85 % / 86 %. |
| C  ablation | Removing every ZCR/RMS/centroid/bandwidth/roll-off/flatness descriptor: Tamil bal-acc 0.992 → **0.980**, Malayalam 1.000 → **1.000** — cue is distributed, not a single feature. |
| D  matched condition | Tamil has **no** matched condition (0 depressed @48 k, 0 non-depressed @16 k). Malayalam 48 kHz-only subset (284 D vs 900 ND): LR group-CV bal-acc **1.000** on both resample-only and normalised features → even "matched" sample rate does not remove separability. |
| E  leakage | 0 MD5 overlap, 0 filename overlap train↔test; 1 duplicate MD5 inside train. No train/test leakage — the shortcut is in the *data collection*, not a split bug. |
| leak test | Within Malayalam-depressed (where sr is *not* tied to class) the 54 features predict **native sr with AUC 0.9998 (resample-only) → 0.9986 (band-limited, windowed)**. Band-limiting alone does not hide the recording chain. |

### 4.3 Check F (new): the label is readable from the *non-speech* frames

Classifier sees only the quietest 15 % of frames (background / room / microphone — no voice). Source-held-out AUC (`results/fig_checkF.png`):

| Audio processing | Malayalam | Tamil |
|---|---|---|
| resample only | **0.986** | **0.948** |
| + band-limit, windows, RMS-norm | 0.783 | 0.914 |
| + noise-floor equalisation (spectral subtraction + common pink-noise floor) | 0.813 | 0.894 |

Normalisation halves the Malayalam leak but does **not** remove it, and noise-floor equalisation does not help further. With speech frames only, AUC stays 0.99 (Malayalam) / 0.90 (Tamil) — speech and channel cannot be separated with this corpus.

### 4.4 Source-held-out evaluation (LOSO) and organiser test (`results/eval_results.json`)

LOSO = each fold holds out one depressed source **and** one non-depressed source (5 folds/language); nuisance-projection `k=10` removes the 10 directions that separate sources *within* a class.

| Features (LR head) | Mal LOSO bal-acc / AUC | Mal organiser-test bal-acc (48 k "matched" slice) | Tam LOSO AUC | Tam→Mal AUC | Mal→Tam AUC |
|---|---|---|---|---|---|
| HAND54 | 0.983 / 0.999 | 1.000 (1.000) | 0.944 | 0.902 | 0.800 |
| WavLM-base L6 | 0.999 / 1.000 | 1.000 (1.000) | 0.966 | 1.000 | 0.999 |
| XLSR-53 L12 | 0.997 / 1.000 | 1.000 (1.000) | 0.988 | 1.000 | 1.000 |
| ECAPA-TDNN (192-d) | 0.950 / 0.993 | 0.995 (0.995) | 0.934 | 1.000 | 0.996 |

(Full grid incl. SVM, nuisance projection, ECAPA, oracle-best layer: `results/results_table.md`.)
Layer scan (`results/fig_layerscan.png`): **every layer** of WavLM (min AUC 1.000) and XLSR-53 (min 0.998) already separates the classes — even layer 1, which is essentially signal-level.

### 4.5 ECAPA-TDNN note
ECAPA (192-d, SpeechBrain VoxCeleb) is trained for speaker identity — exactly the property that encodes the recording environment. It is the *weakest* of the learned representations in source-held-out Malayalam (bal-acc 0.95 LR / 0.85–0.91 SVM+projection, AUC still 0.99) but still ≈ 1.00 on the organiser test and cross-lingual, i.e. it also reads the recording campaign. Row included in the table above and in `results/results_table.md`.

### 4.6 What this means
* High scores are **not** evidence of acoustic depression detection. They are evidence that depressed and non-depressed recordings were collected in different recording set-ups, and that every representation (even a 54-number hand-crafted vector) can read that set-up.
* The first-review conclusion stands, strengthened: *"scores are provisional and not claimed as depression-detection performance."*

---

## 5. What *can* be improved and measured: channel robustness (`results/robustness.json`, `fig_robustness.png`)

Every window — either class, either language — gets an **independently random recording channel** (low-pass 3.5–7.5 kHz, gain ±6 dB, coloured noise at 5–25 dB SNR, 30 % short synthetic reverb, 30 % 8-bit quantisation). Because the channel is independent of the label, a model that survives it is relying less on the recording campaign.

Balanced accuracy on the organiser test set (train→test), clean-trained vs trained with channel augmentation:

| Features | Setting | clean test | random-channel test (clean-trained) | random-channel test (**+ channel-aug training**) |
|---|---|---|---|---|
| HAND54 | Tamil→Tamil | 1.000 | 0.581 | **0.956** |
| HAND54 | Malayalam→Malayalam | 1.000 | 0.746 | **0.995** |
| HAND54 | Tamil→Malayalam | 0.714 | 0.578 | **0.828** |
| HAND54 | Malayalam→Tamil | 0.675 | 0.625 | **0.844** |
| WavLM-L6 | Tamil→Tamil | 1.000 | 0.894 | **1.000** |
| WavLM-L6 | Malayalam→Malayalam | 1.000 | 0.877 | **0.990** |
| WavLM-L6 | Tamil→Malayalam | 1.000 | 0.838 | **0.985** |
| WavLM-L6 | Malayalam→Tamil | 0.894 | 0.812 | **0.894** |

Take-aways: (i) SSL features are far more robust than hand-crafted ones; (ii) channel-randomisation training is a cheap, large, measurable gain; (iii) Malayalam→Tamil is the one remaining weak spot (and Tamil test is itself 100 % sr-confounded, so treat with care).

---

## 6. Final model and demo

* Model: normalise → ≤3×1.5 s windows → WavLM-base layer-6 mean-pool → StandardScaler + LogisticRegression, trained on Tamil+Malayalam train windows **with** channel-randomisation augmentation (`10_final_model.py`).
* Organiser test (clip level, `results/final_model_test.json`): Malayalam 1.000 / 0.995 (clean / random-channel) balanced accuracy, Tamil 1.000 / 1.000.
* Use: `python3 final_pipeline/predict.py clip.wav` → `{"depression_risk": 0.99, "label": "depressed", ...}`. Existing demo code (`web_interface/`, `streamlit_app.py`) can call `predict.predict(path)`.
* **Always show the disclaimer**: screening aid, trained on a corpus with known recording-protocol confounds.

### Limitations to state openly
1. **Speaker-disjoint splits cannot be verified**: non-depressed file names carry no speaker ID; depressed IDs are session-level. We use *source-disjoint* folds as the strongest available proxy.
2. **Condition-matched evaluation is impossible for Tamil** (no overlap in sample rate between classes) and only partially possible for Malayalam (even the 48 kHz subset is separable, so sources differ in more than sample rate).
3. Frozen probes, not full fine-tuning, so far.
4. 1.5 s windows lose long-range prosody (speaking rate, pause structure) that clinical studies emphasise.

---

## 7. What to say in the final review (suggested storyline)
1. First review: curated corpus, features, baselines, **found the shortcut**.
2. Since then: built the remediation toolkit and re-ran the audit — **the shortcut is a recording-campaign effect that survives every preprocessing fix we could apply (Check F is the decisive evidence)**.
3. Benchmarked ECAPA / WavLM / XLSR-53 with source-held-out and cross-lingual protocols — all ≈ 1.00, hence uninformative; we show *why* instead of reporting 100 %.
4. Introduced a **robustness benchmark** and **channel-randomisation training**; delivered a working end-to-end scorer.
5. Recommendation / future work: collect a small matched-condition set (same phone/room/sample rate for both classes, ideally the same speakers) — the one thing that would turn this into a real validation.

---

## 8. The last 5 days: improving performance & fine-tuning

Goal for the remaining time: **maximise robust accuracy and show it with honest protocols**, not push an already-saturated clean score.

| Day | Work | Success criterion |
|---|---|---|
| **1** | Finish ECAPA row; add **HuBERT-large** and **WavLM-large** probes; per-layer *weighted-sum* probe (SUPERB-style) instead of a single layer | Robust-test bal-acc ≥ current WavLM-L6 (0.99 Mal / 1.00 Tam); table complete |
| **2** | **Full fine-tuning of WavLM-base** (last 4 layers + head, lr 1e-5, MPS, 1.5 s windows) **with channel-randomisation + SpecAugment [12]**; early-stop on LOSO fold | ≥ 0.90 on *Malayalam→Tamil* random-channel (currently 0.894) without hurting others |
| **3** | **Gradient-reversal / adversarial source-invariance**: add a head predicting the recording source (D_A/D_F/D_S/ND1–5 within class) with reversed gradient; sweep λ ∈ {0.1, 0.3, 1} | Check F **non-speech AUC drops toward 0.5** on the learned representation while class AUC on LOSO stays high — this is the first measurable de-confounding |
| **4** | **Harder channel augmentation**: real codecs (AMR/MP3/Opus via ffmpeg), real noise (MUSAN-style), device IR convolution, bandwidth 8 kHz↔16 kHz↔48 kHz mixes; test on held-out *augmentation family* (leave-one-augmentation-out) | Robust accuracy under unseen augmentations ≥ 0.9 |
| **5** | Calibration (temperature scaling / threshold tuning with `threshold_tuner.py`), confidence-aware output, error analysis by source, demo wiring, final slides (reuse figures in `results/`) | Report: ECE, per-source error table, final figures; slides + demo rehearsed |

Stretch (only if Days 1–4 finish early): longer context (3–5 s windows, attention pooling), Tamil/Malayalam pronunciation-aware features, test-time augmentation.

Experiments to **avoid** (they will only produce misleading 100 % numbers): random stratified CV on clip-level features, any train/test pair from the same recording campaign, reporting the organiser-test score without the robustness table next to it.

---

## 9. Repo housekeeping
* The 7.7 GB `Group-15A.zip` and the two organiser test zips are **git-ignored** (`*.zip`); GitHub rejects files > 100 MB and the push kept failing with HTTP 408. Share them via Drive.
* `final_pipeline/cache/` (≈ 2 GB of windows/embeddings) is git-ignored; it regenerates with `run_all.sh`.
* Large model checkpoints in `checkpoints/`, `final_models/`, `outputs/` (≈ 30 GB) are not for Git — use Git LFS or a release asset if they must be shared.
