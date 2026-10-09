"""Stage 4: corpus audit Checks A-E (before) + post-remediation leak tests (after)."""
import json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from common import *
from eval_utils import *

man = pd.read_csv(f"{CACHE}/manifest.csv")
R = {}

# ---- Check A: descriptive stats ------------------------------------------------
hf = {t: pd.read_csv(f"{CACHE}/handfeat_{t}.csv") for t in ["resamp", "norm"]}
clipf = {t: d.groupby("clip").mean(numeric_only=True).drop(columns="win") for t, d in hf.items()}
tr = man[man.split == "train"]
A = {}
for lang in ["Tamil", "Malayalam"]:
    for y in [0, 1]:
        s = tr[(tr.lang == lang) & (tr.y == y)]
        A[f"{lang}_y{y}"] = dict(n=len(s), dur_mean=s.duration.mean(), dur_std=s.duration.std(),
                                 zcr_resamp=clipf["resamp"].loc[s.index, "zcr_mean"].mean(),
                                 zcr_norm=clipf["norm"].loc[s.index, "zcr_mean"].mean(),
                                 frac_16k=float((s.sr == 16000).mean()))
R["A_descriptives"] = A

# ---- Check B: metadata-only control (native sr, duration) -----------------------
B = {}
for lang in ["Tamil", "Malayalam"]:
    s = tr[tr.lang == lang]; t = man[(man.split == "test") & (man.lang == lang)]
    for name, cols in [("sr_only", ["sr"]), ("duration_only", ["duration"]), ("sr+duration", ["sr", "duration"])]:
        clf = GradientBoostingClassifier(random_state=0).fit(s[cols], s.y)
        cv = cross_val_score(GradientBoostingClassifier(random_state=0), s[cols], s.y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="accuracy")
        B[f"{lang}_{name}"] = dict(cv_acc=float(cv.mean()), test_acc=float((clf.predict(t[cols]) == t.y).mean()))
R["B_metadata_only"] = B

# ---- Check C: ablation of metadata-correlated descriptors (resample-only audio) --
meta_corr = [c for c in hf["resamp"].columns if c.startswith(("zcr", "rolloff", "centroid", "bandwidth", "flatness", "rms"))]
C = {}
for lang in ["Tamil", "Malayalam"]:
    s = tr[tr.lang == lang]; idx = s.index
    for name, drop in [("all_features", []), ("minus_metadata_correlated", meta_corr)]:
        X = clipf["resamp"].loc[idx].drop(columns=drop)
        cv = cross_val_score(make("LR"), X, s.y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="balanced_accuracy")
        C[f"{lang}_{name}"] = float(cv.mean())
R["C_ablation_bal_acc_LR_5foldCV"] = C

# ---- Check D: matched-condition subset (Malayalam, 48 kHz only) -----------------
d = tr[(tr.lang == "Malayalam") & (tr.sr == 48000)]
D = {"n_depressed_48k": int((d.y == 1).sum()), "n_nondepressed_48k": int((d.y == 0).sum()),
     "tamil_has_matched_condition": bool(tr[(tr.lang == "Tamil")].groupby("y").sr.nunique().min() > 1)}
for t in ["resamp", "norm"]:
    X = clipf[t].loc[d.index]
    cv = cross_val_score(make("LR"), X, d.y, cv=GroupKFold(5), groups=d.group, scoring="balanced_accuracy")
    D[f"LR_groupcv_bal_acc_{t}"] = float(cv.mean())
R["D_matched_condition"] = D

# ---- Check E: leakage ---------------------------------------------------------
te = man[man.split == "test"]
R["E_leakage"] = dict(md5_overlap=int(len(set(tr.md5) & set(te.md5))), filename_overlap_within_language=int(
    sum(len(set(tr[tr.lang == l].file) & set(te[te.lang == l].file)) for l in ["Tamil", "Malayalam"])),
    duplicate_md5_inside_train=int(tr.md5.duplicated().sum()))

# ---- Post-remediation leak test: can features predict NATIVE SAMPLE RATE? ---------
# Malayalam depressed clips exist at both 16k and 48k -> sr is not confounded with class there.
m = tr[(tr.lang == "Malayalam") & (tr.y == 1)]
L = {}
for t in ["resamp", "norm"]:
    X = clipf[t].loc[m.index]; ysr = (m.sr == 48000).astype(int)
    for mn in ["LR", "RF"]:
        cv = cross_val_score(make(mn), X, ysr, cv=GroupKFold(5), groups=m.group, scoring="roc_auc")
        L[f"{t}_{mn}_predict_native_sr_auc"] = float(cv.mean())
R["leak_test_handcrafted"] = L
json.dump(R, open(f"{RES}/audit.json", "w"), indent=2, default=float)
print(json.dumps(R, indent=1, default=float))
