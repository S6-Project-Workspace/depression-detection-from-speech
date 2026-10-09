"""Check F (new): non-speech control. Classify the label from ONLY the quietest 15% of frames
(background noise / room / microphone), which contain no voice. AUC well above 0.5 means the
label is readable from the recording environment alone."""
import json, numpy as np, pandas as pd, librosa
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from common import *
from eval_utils import make, score

man = pd.read_csv(f"{CACHE}/manifest.csv")
ts = pd.read_csv(f"{CACHE}/train_sources.csv")
key = {(l, f, y): s for l, f, y, s in zip(ts.lang, ts.file, ts.y, ts.src)}
man["src"] = [key.get((l, f, y), "test") if sp == "train" else "test" for l, f, y, sp in zip(man.lang, man.file, man.y, man.split)]

def parts(a):
    M = librosa.power_to_db(librosa.feature.melspectrogram(y=a, sr=SR, n_fft=512, hop_length=160, n_mels=40) + 1e-10)
    e = M.mean(0); lo = e <= np.quantile(e, 0.15); hi = e >= np.quantile(e, 0.70)
    return np.concatenate([M[:, lo].mean(1), M[:, hi].mean(1)])   # [noise-floor 40 | speech 40]

def one(i, path, variant):
    a, sr = load_raw(path)
    if variant == "raw16k":
        a = librosa.resample(a.astype(np.float64), orig_sr=sr, target_sr=SR, res_type="soxr_hq").astype(np.float32) if sr != SR else a
    elif variant == "norm_v1":
        a = normalise(a, sr, True)
    else:
        a = normalise_v2(a, sr)
    return parts(a)

PAIRS = [("D_A", "ND1"), ("D_F", "ND2"), ("D_S", "ND3"), ("D_A", "ND4"), ("D_F", "ND5")]
def loso_auc(X, lang):
    m = (man.lang == lang) & (man.split == "train"); idx = man.index[m]; s = {}
    for d, n in PAIRS:
        te = idx[man.loc[idx, "src"].isin([d, n])]; tr = idx.difference(te)
        mod = make("LR").fit(X[tr], man.y[tr]); s.update(dict(zip(te, score(mod, X[te]))))
    s = pd.Series(s); return roc_auc_score(man.y[s.index], s)

if __name__ == "__main__":
    out = {}
    for variant in ["raw16k", "norm_v1", "norm_v2_nfe"]:
        X = np.stack(Parallel(n_jobs=8)(delayed(one)(i, p, variant) for i, p in zip(man.index, man.path)))
        np.save(f"{CACHE}/checkF_{variant}.npy", X)
        for lang in ["Malayalam", "Tamil"]:
            out[f"{variant}|{lang}|noise_floor_only_AUC"] = loso_auc(X[:, :40], lang)
            out[f"{variant}|{lang}|speech_frames_only_AUC"] = loso_auc(X[:, 40:], lang)
        print(variant, {k.split("|", 1)[1]: round(v, 3) for k, v in out.items() if k.startswith(variant)}, flush=True)
        json.dump(out, open(f"{RES}/checkF.json", "w"), indent=1)
