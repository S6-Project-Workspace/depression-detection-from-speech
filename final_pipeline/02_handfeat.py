"""Stage 2: 54 hand-crafted descriptors per 1.5 s window
(prosodic 3, time-domain 4, spectral 8 [+zcr], cepstral 39)."""
import numpy as np, pandas as pd, librosa
from joblib import Parallel, delayed
from common import *

NAMES = None
def feats(w):
    f = {}
    f0 = librosa.yin(w, fmin=70, fmax=400, sr=SR, frame_length=1024)
    f["f0_mean"], f["f0_std"], f["f0_range"] = float(np.mean(f0)), float(np.std(f0)), float(np.ptp(f0))
    z = librosa.feature.zero_crossing_rate(w)[0]; r = librosa.feature.rms(y=w)[0]
    f.update(zcr_mean=z.mean(), zcr_std=z.std(), rms_mean=r.mean(), rms_std=r.std())
    for n, x in [("centroid", librosa.feature.spectral_centroid(y=w, sr=SR)[0]),
                 ("bandwidth", librosa.feature.spectral_bandwidth(y=w, sr=SR)[0]),
                 ("rolloff", librosa.feature.spectral_rolloff(y=w, sr=SR)[0]),
                 ("flatness", librosa.feature.spectral_flatness(y=w)[0])]:
        f[n + "_mean"], f[n + "_std"] = x.mean(), x.std()
    m = librosa.feature.mfcc(y=w, sr=SR, n_mfcc=13); d = librosa.feature.delta(m)
    for i in range(13):
        f[f"mfcc{i}_mean"], f[f"mfcc{i}_std"], f[f"dmfcc{i}_mean"] = m[i].mean(), m[i].std(), d[i].mean()
    return {k: float(v) for k, v in f.items()}

if __name__ == "__main__":
    for tag in ["norm", "resamp"]:
        W = np.load(f"{CACHE}/windows_{tag}.npy")
        rows = Parallel(n_jobs=8, batch_size=64)(delayed(feats)(w) for w in W)
        df = pd.DataFrame(rows)
        df = pd.concat([pd.read_csv(f"{CACHE}/windows_{tag}_index.csv"), df], axis=1)
        df.to_csv(f"{CACHE}/handfeat_{tag}.csv", index=False)
        print(tag, df.shape)
