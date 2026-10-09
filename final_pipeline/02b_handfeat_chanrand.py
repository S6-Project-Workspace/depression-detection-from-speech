import numpy as np, pandas as pd
from joblib import Parallel, delayed
from common import *
import importlib; hf = importlib.import_module("02_handfeat")
W = np.load(f"{CACHE}/windows_chanrand.npy")
rows = Parallel(n_jobs=6, batch_size=64)(delayed(hf.feats)(w) for w in W)
pd.concat([pd.read_csv(f"{CACHE}/windows_norm_index.csv"), pd.DataFrame(rows)], axis=1).to_csv(f"{CACHE}/handfeat_chanrand.csv", index=False); print("ok")
