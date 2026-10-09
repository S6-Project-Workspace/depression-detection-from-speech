"""Collect per-file container metadata (native sr, channels, subtype, duration, md5)
for every train and test clip. Output: audit/metadata.csv"""
import hashlib, glob, os, sys
import pandas as pd, soundfile as sf
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = [
    ("Tamil", "Depressed", "NLP Dataset/Tamil/Depressed/Train_set"),
    ("Tamil", "Non-depressed", "NLP Dataset/Tamil/Non-depressed/Train_set"),
    ("Malayalam", "Depressed", "NLP Dataset/Malayalam/Depressed/Train_set"),
    ("Malayalam", "Non-depressed", "NLP Dataset/Malayalam/Non_depressed/Train_set"),
]
TEST = [("Tamil", "Test Set/tamil_test/Test-set-tamil"), ("Malayalam", "Test Set/malayalam_test/Test_set_mal")]

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def info(path, lang, label, split):
    i = sf.info(path)
    return dict(path=os.path.relpath(path, ROOT), file=os.path.basename(path), lang=lang, label=label, split=split,
                sr=i.samplerate, channels=i.channels, subtype=i.subtype, duration=i.duration, md5=md5(path))

def main():
    jobs = []
    for lang, lab, d in TRAIN:
        for p in sorted(glob.glob(os.path.join(ROOT, d, "*.wav"))):
            jobs.append((p, lang, lab, "train"))
    for lang, d in TEST:
        for p in sorted(glob.glob(os.path.join(ROOT, d, "**", "*.wav"), recursive=True)):
            jobs.append((p, lang, None, "test"))
    rows = Parallel(n_jobs=8)(delayed(info)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    # test labels come from the ground-truth sheets
    for lang in ["Tamil", "Malayalam"]:
        gt = pd.read_excel(os.path.join(ROOT, "Test Set", f"{lang}_GT.xlsx"))
        print(lang, "GT columns:", list(gt.columns), len(gt))
    df.to_csv(os.path.join(ROOT, "audit", "metadata.csv"), index=False)
    print(df.groupby(["split", "lang", "label"], dropna=False).size())
    print(df.groupby(["lang", "label", "sr"], dropna=False).size())

if __name__ == "__main__":
    main()
