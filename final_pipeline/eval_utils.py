import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

def make(name):
    if name == "LR":  return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    if name == "SVM": return make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", class_weight="balanced"))
    if name == "RF":  return RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=8, random_state=0)
    raise ValueError(name)

def score(model, X):
    """signed score, >0 => class 1"""
    return model.predict_proba(X)[:, 1] - 0.5 if hasattr(model, "predict_proba") and not hasattr(model, "decision_function") or isinstance(model, RandomForestClassifier) else model.decision_function(X)

def clip_agg(s, clip):
    return pd.Series(s).groupby(np.asarray(clip)).mean()

def metrics(y, s):
    p = (np.asarray(s) > 0).astype(int); y = np.asarray(y)
    out = dict(n=int(len(y)), acc=accuracy_score(y, p), bal_acc=balanced_accuracy_score(y, p),
               macro_f1=f1_score(y, p, average="macro"), cm=confusion_matrix(y, p, labels=[0, 1]).tolist())
    out["auc"] = roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")
    return out

def fit_eval(model_name, Xtr, ytr, Xte, ytest, clip_te):
    """train on windows, predict clip-level (mean window score)."""
    m = make(model_name).fit(Xtr, ytr)
    s = clip_agg(score(m, Xte), clip_te)
    return s

def group_cv(model_name, X, y_win, clip_win, y_clip, group_clip, n_splits=5):
    """clip-level GroupKFold; returns clip-level out-of-fold scores (Series indexed by clip id)."""
    clips = np.array(sorted(set(clip_win)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = {}
    for tr_c, va_c in gkf.split(clips, groups=[group_clip[c] for c in clips]):
        trs, vas = set(clips[tr_c]), set(clips[va_c])
        mtr = np.array([c in trs for c in clip_win]); mva = ~mtr & np.array([c in vas for c in clip_win])
        s = fit_eval(model_name, X[mtr], y_win[mtr], X[mva], None, clip_win[mva])
        oof.update(s.to_dict())
    return pd.Series(oof)
