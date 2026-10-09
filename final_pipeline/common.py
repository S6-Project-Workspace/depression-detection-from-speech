"""Shared helpers for the post-first-review pipeline.

Remediation of the confounds found in the first review:
  * native sample rate  -> every clip is resampled to 16 kHz and band-limited to the
    same telephone band (80 Hz - 3.8 kHz) so 16 kHz and 48 kHz originals are
    spectrally indistinguishable above the cut-off;
  * clip duration       -> classification units are fixed-length 1.5 s windows
    (max 3 per clip, evenly spaced); clip-level decisions average window scores;
  * loudness            -> active-segment RMS normalisation.
"""
import os, re, numpy as np, pandas as pd, soundfile as sf, librosa
from scipy.signal import butter, sosfiltfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "final_pipeline", "cache")
RES = os.path.join(ROOT, "final_pipeline", "results")
SR = 16000
WIN = int(1.5 * SR)
MAX_WIN = 3
LOW, HIGH = 80.0, 3800.0
_SOS = butter(8, [LOW, HIGH], btype="bandpass", fs=SR, output="sos")


def group_id(row):
    """Best-effort recording-session proxy (speaker IDs are not in the file names)."""
    s = re.sub(r"\.wav$", "", row["file"])
    if row["split"] == "test":
        return "test_" + s
    if row["label"] == "Depressed":
        m = re.match(r"D_(A\d+)_(\d+)(?:-\d+)?$", s)
        if m:
            return f"{row['lang']}_{m.group(1)}_{m.group(2)}" if row["lang"] == "Tamil" else f"{row['lang']}_{m.group(1)}"
    return f"{row['lang']}_{s}"  # non-depressed clips are anonymous: one group per file


def manifest():
    df = pd.read_csv(os.path.join(ROOT, "audit", "metadata.csv"))
    y = {}
    for lang, col in [("Tamil", "label"), ("Malayalam", "Label")]:
        gt = pd.read_excel(os.path.join(ROOT, "Test Set", f"{lang}_GT.xlsx"))
        for fn, lab in zip(gt["filename"], gt[col]):
            y[(lang, fn)] = 1 if str(lab).upper().startswith("D") else 0
    df["y"] = [1 if l == "Depressed" else 0 if l == "Non-depressed" else y[(lg, f)]
               for l, lg, f in zip(df.label, df.lang, df.file)]
    df["group"] = df.apply(group_id, axis=1)
    return df.reset_index(drop=True)


def load_raw(path):
    a, sr = sf.read(os.path.join(ROOT, path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    return a, sr


def normalise(a, sr, bandlimit=True):
    if sr != SR:
        a = librosa.resample(a.astype(np.float64), orig_sr=sr, target_sr=SR, res_type="soxr_hq").astype(np.float32)
    if bandlimit:
        a = sosfiltfilt(_SOS, a).astype(np.float32)
    a, _ = librosa.effects.trim(a, top_db=35)
    if len(a) < 256:
        a = np.pad(a, (0, 256 - len(a)))
    act = a[np.abs(a) > 0.1 * np.abs(a).max()]
    rms = np.sqrt(np.mean(act ** 2)) + 1e-8
    return np.clip(a * (0.05 / rms), -1, 1).astype(np.float32)


def windows(a):
    if len(a) <= WIN:
        return [np.pad(a, (0, WIN - len(a)), mode="reflect" if len(a) > 1 else "constant")] if len(a) < WIN else [a]
    n = min(MAX_WIN, 1 + (len(a) - WIN) // (WIN // 2))
    starts = np.linspace(0, len(a) - WIN, n).astype(int)
    return [a[s:s + WIN] for s in starts]


def noise_floor_equalise(a, floor_db=-48.0, alpha=1.5, spec_floor=0.05, seed=0):
    """Remediation v2: (1) estimate each clip's stationary noise spectrum from its quietest
    10% frames and Wiener/spectral-subtract it; (2) add the SAME synthetic pink-ish noise floor
    to every clip, so any residual noise difference between recording campaigns is masked
    by a common floor instead of encoding the label."""
    S = librosa.stft(a, n_fft=512, hop_length=160)
    P = np.abs(S) ** 2
    e = P.sum(0); q = e <= np.quantile(e, 0.10)
    N = P[:, q].mean(1, keepdims=True)
    G = np.maximum(1.0 - alpha * N / (P + 1e-12), spec_floor)
    den = librosa.istft(S * np.sqrt(G), hop_length=160, length=len(a)).astype(np.float32)
    rng = np.random.default_rng(seed)
    wn = rng.standard_normal(len(den)).astype(np.float32)
    f = np.fft.rfft(wn); k = np.arange(len(f)); f /= np.sqrt(np.maximum(k, 1)); pn = np.fft.irfft(f, len(den)).astype(np.float32)
    pn *= 10 ** (floor_db / 20) / (pn.std() + 1e-9)
    return den + pn


def normalise_v2(a, sr):
    """resample -> noise-floor equalisation -> band-limit -> trim -> RMS normalise."""
    if sr != SR:
        a = librosa.resample(a.astype(np.float64), orig_sr=sr, target_sr=SR, res_type="soxr_hq").astype(np.float32)
    a = a / (np.abs(a).max() + 1e-8) * 0.5
    a = noise_floor_equalise(a)
    return normalise(a, SR, bandlimit=True)
