"""Stage 10: final end-to-end model = normalise -> 1.5 s windows -> WavLM-base layer-6 mean-pool
-> StandardScaler + logistic regression, trained on Tamil+Malayalam train windows with
channel-randomisation augmentation. Saves model/wavlm_l6_lr.joblib and evaluates on the organiser test sets."""
import json, joblib, numpy as np, pandas as pd, sys
sys.argv = ["x"]
exec(open("05_evaluate.py").read().split('if __name__')[0])
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
L = 6
Xc = np.load(f"{CACHE}/emb_wavlm-base-plus_norm.npy")[:, L].astype(np.float32); Xp = np.load(f"{CACHE}/emb_wavlm-base-plus_chanrand.npy")[:, L].astype(np.float32)
trm = (w["split"] == "train").values
X = np.vstack([Xc[trm], Xp[trm]]); y = np.concatenate([w["y"].values[trm]] * 2)
model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")).fit(X, y)
joblib.dump(model, "model/wavlm_l6_lr.joblib")
R = {}
for lang in ["Malayalam", "Tamil"]:
    for view, Z in [("clean", Xc), ("chanrand", Xp)]:
        te = ((w["lang"] == lang) & (w["split"] == "test")).values
        s = pd.Series(model.decision_function(Z[te])).groupby(wclip[te]).mean(); mt = man.loc[s.index]
        R[f"{lang}_{view}"] = metrics(mt["y"], s)
        print(lang, view, {k: round(v, 3) for k, v in R[f"{lang}_{view}"].items() if k in ("acc", "bal_acc", "macro_f1", "auc")})
json.dump(R, open(f"{RES}/final_model_test.json", "w"), indent=1, default=float)
