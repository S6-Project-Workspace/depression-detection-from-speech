"""
Baseline classifier training + evaluation, plus a diagnostic that
quantifies the native-sample-rate / label leakage found in this corpus.
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
)

df = pd.read_csv("baseline_features.csv")
df = df[df["_error"].isna()].copy()

feature_cols = [c for c in df.columns if c not in
                 ["_error", "filepath", "filename", "label", "language", "split", "native_sr"]]

train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)

print(f"Train: {len(train_df)}  Test: {len(test_df)}")
print(f"Feature count: {len(feature_cols)}")

results = {}

# ------------------------------------------------------------------
# Diagnostic: trivial "sample-rate-only" classifier
# (native_sr >= 32000 -> predict Non-depressed, else Depressed)
# quantifies how much signal is available from metadata alone
# ------------------------------------------------------------------
def sr_only_predict(sr_series):
    return (sr_series < 32000).astype(int)  # 1 = depressed

for name, d in [("train", train_df), ("test", test_df)]:
    pred = sr_only_predict(d["native_sr"])
    y = d["label"].values
    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    results[f"sr_only_{name}"] = {"accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1}
    print(f"\n[Sample-rate-only diagnostic — {name}] acc={acc:.4f} macroF1={f1:.4f}")

    for lang in d["language"].unique():
        sub = d[d["language"] == lang]
        pred_l = sr_only_predict(sub["native_sr"])
        acc_l = accuracy_score(sub["label"], pred_l)
        p_l, r_l, f1_l, _ = precision_recall_fscore_support(sub["label"], pred_l, average="macro", zero_division=0)
        results[f"sr_only_{name}_{lang}"] = {"accuracy": acc_l, "macro_f1": f1_l}
        print(f"    {lang}: acc={acc_l:.4f} macroF1={f1_l:.4f}  (n={len(sub)})")

# ------------------------------------------------------------------
# Real baseline: engineered acoustic features -> classifier
# (audio was resampled properly, so native_sr itself is EXCLUDED
#  from the feature set used by the real classifier)
# ------------------------------------------------------------------
X_train = train_df[feature_cols].values
y_train = train_df["label"].values
X_test = test_df[feature_cols].values
y_test = test_df["label"].values

models = {
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)),
    ]),
    "SVM-RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=2.0, class_weight="balanced", probability=True, random_state=42)),
    ]),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, pipe in models.items():
    print(f"\n=== {model_name} ===")
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
    print(f"5-fold CV macro-F1 on train: {cv_scores.mean():.4f} +- {cv_scores.std():.4f}")
    results[f"{model_name}_cv"] = {"mean_f1": cv_scores.mean(), "std_f1": cv_scores.std(), "folds": cv_scores.tolist()}

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    print(f"TEST SET: acc={acc:.4f}  macroP={p:.4f}  macroR={r:.4f}  macroF1={f1:.4f}")
    print("Confusion matrix [[TN,FP],[FN,TP]]:", cm)
    print(classification_report(y_test, y_pred, target_names=["Non-depressed", "Depressed"], zero_division=0))

    results[f"{model_name}_test"] = {
        "accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1,
        "confusion_matrix": cm,
    }

    # per-language breakdown
    for lang in test_df["language"].unique():
        mask = (test_df["language"] == lang).values
        p_l, r_l, f1_l, _ = precision_recall_fscore_support(y_test[mask], y_pred[mask], average="macro", zero_division=0)
        acc_l = accuracy_score(y_test[mask], y_pred[mask])
        results[f"{model_name}_test_{lang}"] = {"accuracy": acc_l, "macro_f1": f1_l}
        print(f"    {lang}: acc={acc_l:.4f} macroF1={f1_l:.4f}  (n={mask.sum()})")

with open("baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nWrote baseline_results.json")
