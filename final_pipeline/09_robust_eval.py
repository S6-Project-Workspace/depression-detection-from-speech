"""Stage 9: clean vs channel-randomised test, with and without channel-randomisation augmentation."""
import json, numpy as np, pandas as pd, sys
sys.argv = ["x"]
exec(open("05_evaluate.py").read().split('if __name__')[0])   # reuse manifest/windows/fit_score/loso helpers

def feats(kind):
    if kind == "WavLM-L6":
        return (np.load(f"{CACHE}/emb_wavlm-base-plus_norm.npy")[:, 6].astype(np.float32),
                np.load(f"{CACHE}/emb_wavlm-base-plus_chanrand.npy")[:, 6].astype(np.float32))
    import subprocess
    hf = lambda t: pd.read_csv(f"{CACHE}/handfeat_{t}.csv").drop(columns=["clip", "win"]).values.astype(np.float32)
    return hf("norm"), hf("chanrand")

def run(kind, train_lang, test_lang, aug, test_view, model="LR"):
    Xc, Xp = feats(kind)
    trm = ((w["lang"] == train_lang) & (w["split"] == "train")).values
    te = ((w["lang"] == test_lang) & (w["split"] == "test")).values
    Xtr = np.vstack([Xc[trm], Xp[trm]]) if aug else Xc[trm]
    ytr = np.concatenate([w["y"].values[trm]] * (2 if aug else 1)); str_ = np.concatenate([w["src"].values[trm]] * (2 if aug else 1))
    Xte = (Xp if test_view == "chanrand" else Xc)[te]
    s = fit_score(model, Xtr, ytr, str_, Xte, 0); cs = clip_agg(s, wclip[te]); m = metrics(man.loc[cs.index, "y"], cs)
    return dict(auc=m["auc"], bal_acc=m["bal_acc"], macro_f1=m["macro_f1"])

if __name__ == "__main__":
    kinds = ["WavLM-L6"] + (["HAND54"] if __import__("os").path.exists(f"{CACHE}/handfeat_chanrand.csv") else [])
    R = []
    for kind in kinds:
        for tl, sl in [("Malayalam", "Malayalam"), ("Tamil", "Tamil"), ("Tamil", "Malayalam"), ("Malayalam", "Tamil")]:
            for aug in (False, True):
                for view in ("clean", "chanrand"):
                    r = run(kind, tl, sl, aug, view); r.update(features=kind, train=tl, test=sl, augmented_training=aug, test_view=view); R.append(r)
                    print(kind, f"{tl}->{sl}", "aug" if aug else "clean-train", view, "auc=%.3f bal=%.3f" % (r["auc"], r["bal_acc"]), flush=True)
    json.dump(R, open(f"{RES}/robustness.json", "w"), indent=1)
