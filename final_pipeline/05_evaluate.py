"""Stage 5: source-aware evaluation of hand-crafted + SSL features.

Protocols (all metrics clip-level, mean of window scores):
  LOSO   leave-one-recording-source-out (held-out folds contain a depressed source AND a
         non-depressed source never seen in training) -- the primary, confound-aware estimate
  TEST   organiser's held-out test set (reported with the 48 kHz 'matched' slice)
  XLING  train on one language, test on the other language's test set
Optional nuisance-subspace projection (WCCN/INLP-style): remove the top-k directions that
separate recording sources *within* a class, learned on training folds only.
"""
import sys, json, itertools, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from common import *
from eval_utils import make, score, clip_agg, metrics

man = pd.read_csv(f"{CACHE}/manifest.csv")
tr_idx = man.index[man.split == "train"]
# train sources keyed by (lang,file)
ts = pd.read_csv(f"{CACHE}/train_sources.csv")
key = {(l, f, y): s for l, f, y, s in zip(ts.lang, ts.file, ts.y, ts.src)}
man["src"] = [key.get((l, f, y), "test") if sp == "train" else "test" for l, f, y, sp in zip(man.lang, man.file, man.y, man.split)]
widx = pd.read_csv(f"{CACHE}/windows_norm_index.csv")
wclip = widx["clip"].values
w = man.loc[wclip].reset_index(drop=True)

def LOSO_pairs(lang):
    return [("D_A", "ND1"), ("D_F", "ND2"), ("D_S", "ND3"), ("D_A", "ND4"), ("D_F", "ND5")]

def nuisance_basis(X, y, s, k):
    """top-k directions of between-source scatter *within class* (cannot encode class)."""
    if k == 0: return None
    devs = []
    for c in (0, 1):
        m = y == c; mu = X[m].mean(0)
        for ss in np.unique(s[m]):
            n = (m & (s == ss)).sum()
            devs.append(np.sqrt(n) * (X[m & (s == ss)].mean(0) - mu))
    _, _, Vt = np.linalg.svd(np.stack(devs), full_matrices=False)
    return Vt[:k].T

def fit_score(model, Xtr, ytr, str_, Xte, k):
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
    V = nuisance_basis(A, ytr, str_, k)
    if V is not None: A, B = A - A @ V @ V.T, B - B @ V @ V.T
    m = make(model)
    # model pipelines include their own scaler; harmless after projection
    m.fit(A, ytr)
    return score(m, B)

def loso(X, lang, model="LR", k=0):
    mask_l = (w["lang"] == lang).values & (w["split"] == "train").values
    oof, folds = {}, []
    for dsrc, nsrc in LOSO_pairs(lang):
        te = mask_l & w["src"].isin([dsrc, nsrc]).values
        trm = mask_l & ~te
        s = fit_score(model, X[trm], w["y"].values[trm], w["src"].values[trm], X[te], k)
        cs = clip_agg(s, wclip[te]); y = man.loc[cs.index, "y"]
        folds.append(metrics(y, cs)); oof.update(cs.to_dict())
    oof = pd.Series(oof); out = metrics(man.loc[oof.index, "y"], oof)
    out["fold_bal_acc"] = [f["bal_acc"] for f in folds]; out["fold_auc"] = [f["auc"] for f in folds]
    return out

def test_eval(X, train_langs, test_lang, model="LR", k=0):
    trm = (w["lang"].isin(train_langs) & (w["split"] == "train")).values
    te = ((w["lang"] == test_lang) & (w["split"] == "test")).values
    s = fit_score(model, X[trm], w["y"].values[trm], w["src"].values[trm], X[te], k)
    cs = clip_agg(s, wclip[te]); mt = man.loc[cs.index]
    out = metrics(mt.y, cs)
    sl = mt.sr == 48000
    out["slice_48k"] = metrics(mt.y[sl], cs[sl]) if sl.sum() and mt.y[sl].nunique() == 2 else None
    out["acc_16k_depressed_recall"] = float((cs[mt.sr == 16000] > 0).mean()) if (mt.sr == 16000).any() else None
    return out

def load_feats():
    hf = pd.read_csv(f"{CACHE}/handfeat_norm.csv")
    F = {"HAND54": hf.drop(columns=["clip", "win"]).values.astype(np.float32)}
    for nm in ["wavlm-base-plus", "wav2vec2-large-xlsr-53", "hubert-large-ll60k", "ecapa"]:
        p = f"{CACHE}/emb_{nm}_norm.npy"
        try: F[nm] = np.load(p).astype(np.float32)
        except FileNotFoundError: pass
    return F

if __name__ == "__main__":
    F = load_feats(); R = {"layer_scan_malayalam_LOSO_LR": {}, "table": []}
    # ---- layer scan (Malayalam LOSO, LR) -----------------------------------
    best = {}
    for nm, X in F.items():
        if X.ndim == 3:
            sc = []
            for L in range(X.shape[1]):
                o = loso(X[:, L, :], "Malayalam"); sc.append(o["auc"]); print(nm, "layer", L, round(o["auc"], 3), round(o["bal_acc"], 3), flush=True)
            R["layer_scan_malayalam_LOSO_LR"][nm] = sc; best[nm] = (int(np.argmax(sc)), X.shape[1] // 2)
    json.dump(R, open(f"{RES}/eval_layerscan.json", "w"), indent=1)
    # ---- main table --------------------------------------------------------
    for nm, X in F.items():
        views = {"": X} if X.ndim == 2 else {f"@mid(L{best[nm][1]})": X[:, best[nm][1], :], f"@best(L{best[nm][0]})": X[:, best[nm][0], :]}
        for vn, Xv in views.items():
            for model, k in itertools.product(["LR", "SVM"], [0, 10]):
                row = dict(features=nm + vn, model=model, nuisance_k=k)
                row["LOSO_mal"] = loso(Xv, "Malayalam", model, k)
                row["LOSO_tam"] = loso(Xv, "Tamil", model, k)
                row["TEST_mal"] = test_eval(Xv, ["Malayalam"], "Malayalam", model, k)
                row["TEST_tam"] = test_eval(Xv, ["Tamil"], "Tamil", model, k)
                row["XLING_tam2mal"] = test_eval(Xv, ["Tamil"], "Malayalam", model, k)
                row["XLING_mal2tam"] = test_eval(Xv, ["Malayalam"], "Tamil", model, k)
                R["table"].append(row)
                print(row["features"], model, k, "LOSOmal bal=%.3f auc=%.3f | TESTmal bal=%.3f | x-ling t2m auc=%.3f" % (
                    row["LOSO_mal"]["bal_acc"], row["LOSO_mal"]["auc"], row["TEST_mal"]["bal_acc"], row["XLING_tam2mal"]["auc"]), flush=True)
                json.dump(R, open(f"{RES}/eval_results.json", "w"), indent=1, default=float)
