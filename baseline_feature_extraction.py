"""
Baseline acoustic feature extraction for depression detection.

Time-domain (ZCR, short-time energy/RMS), frequency-domain (spectral
centroid/bandwidth/rolloff), and cepstral (MFCC) descriptors, computed
after proper high-quality resampling to 16 kHz mono — deliberately NOT
naive decimation, since native sample rate is confounded with class
label in this corpus (see README_BASELINE_FINDINGS.md).
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

TARGET_SR = 16000
N_MFCC = 13

TRAIN_GROUPS = {
    ("Tamil", "Depressed", 1): "NLP Dataset/Tamil/Depressed/Train_set",
    ("Tamil", "Non-depressed", 0): "NLP Dataset/Tamil/Non-depressed/Train_set",
    ("Malayalam", "Depressed", 1): "NLP Dataset/Malayalam/Depressed/Train_set",
    ("Malayalam", "Non-depressed", 0): "NLP Dataset/Malayalam/Non_depressed/Train_set",
}


def extract_features(path: str) -> dict:
    """Load audio with proper resampling, extract time/freq/cepstral features."""
    try:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=TARGET_SR)
        audio = audio.astype(np.float32)
        if len(audio) < 400:
            audio = np.pad(audio, (0, 400 - len(audio)))

        # time-domain
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        rms = librosa.feature.rms(y=audio)[0]

        # frequency-domain
        cent = librosa.feature.spectral_centroid(y=audio, sr=TARGET_SR)[0]
        bw = librosa.feature.spectral_bandwidth(y=audio, sr=TARGET_SR)[0]
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=TARGET_SR)[0]
        flatness = librosa.feature.spectral_flatness(y=audio)[0]

        # cepstral
        mfcc = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC)
        mfcc_delta = librosa.feature.delta(mfcc)

        feat = {
            "duration": len(audio) / TARGET_SR,
            "native_sr": sr,
            "zcr_mean": float(np.mean(zcr)), "zcr_std": float(np.std(zcr)),
            "rms_mean": float(np.mean(rms)), "rms_std": float(np.std(rms)),
            "centroid_mean": float(np.mean(cent)), "centroid_std": float(np.std(cent)),
            "bandwidth_mean": float(np.mean(bw)), "bandwidth_std": float(np.std(bw)),
            "rolloff_mean": float(np.mean(rolloff)), "rolloff_std": float(np.std(rolloff)),
            "flatness_mean": float(np.mean(flatness)), "flatness_std": float(np.std(flatness)),
        }
        for i in range(N_MFCC):
            feat[f"mfcc{i}_mean"] = float(np.mean(mfcc[i]))
            feat[f"mfcc{i}_std"] = float(np.std(mfcc[i]))
            feat[f"mfcc_delta{i}_mean"] = float(np.mean(mfcc_delta[i]))
        feat["_error"] = None
        return feat
    except Exception as e:
        return {"_error": str(e)}


def process_file(args):
    path, label, lang, split = args
    feat = extract_features(path)
    feat["filepath"] = path
    feat["filename"] = os.path.basename(path)
    feat["label"] = label
    feat["language"] = lang
    feat["split"] = split
    return feat


def build_train_manifest():
    rows = []
    for (lang, cls, label), d in TRAIN_GROUPS.items():
        for f in Path(d).glob("*.wav"):
            rows.append((str(f), label, lang, "train"))
    return rows


def build_test_manifest():
    rows = []
    tamil_gt = pd.read_excel("Test Set/Tamil_GT.xlsx")
    mal_gt = pd.read_excel("Test Set/Malayalam_GT.xlsx")

    def find_file(fn, test_dir):
        for root, _, files in os.walk(test_dir):
            if fn in files:
                return os.path.join(root, fn)
        return None

    for _, row in tamil_gt.iterrows():
        fn = row["filename"]
        label = 1 if str(row["label"]).upper() == "D" else 0
        path = find_file(fn, "Test Set/tamil_test")
        if path:
            rows.append((path, label, "Tamil", "test"))

    for _, row in mal_gt.iterrows():
        fn = row["filename"]
        label = 1 if str(row["Label"]).upper() == "D" else 0
        path = find_file(fn, "Test Set/malayalam_test")
        if path:
            rows.append((path, label, "Malayalam", "test"))

    return rows


def main():
    train_rows = build_train_manifest()
    test_rows = build_test_manifest()
    print(f"Train files: {len(train_rows)}  Test files: {len(test_rows)}")

    all_rows = train_rows + test_rows
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = {ex.submit(process_file, r): r for r in all_rows}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
            results.append(fut.result())

    df = pd.DataFrame(results)
    n_err = df["_error"].notna().sum()
    print(f"Errors: {n_err}")
    if n_err:
        print(df[df["_error"].notna()][["filepath", "_error"]].head(20))

    df.to_csv("baseline_features.csv", index=False)
    print("Wrote baseline_features.csv", df.shape)


if __name__ == "__main__":
    main()
