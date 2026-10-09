"""Stage 1: normalised fixed-length windows for every clip (+ an un-band-limited
'resample-only' variant that reproduces the first-review preprocessing for comparison)."""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from common import *

def one(i, path, bl):
    a, sr = load_raw(path)
    return i, windows(normalise(a, sr, bandlimit=bl))

if __name__ == "__main__":
    df = manifest()
    df.to_csv(f"{CACHE}/manifest.csv", index=False)
    for tag, bl in [("norm", True), ("resamp", False)]:
        out = Parallel(n_jobs=8)(delayed(one)(i, p, bl) for i, p in zip(df.index, df.path))
        W, idx = [], []
        for i, ws in out:
            for k, w in enumerate(ws):
                W.append(w); idx.append((i, k))
        np.save(f"{CACHE}/windows_{tag}.npy", np.stack(W).astype(np.float32))
        pd.DataFrame(idx, columns=["clip", "win"]).to_csv(f"{CACHE}/windows_{tag}_index.csv", index=False)
        print(tag, len(W), "windows from", len(df), "clips")
    print(df.groupby(["split", "lang", "y"]).size())
