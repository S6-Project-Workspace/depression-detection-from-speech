"""Score one audio file:  python predict.py path/to/clip.wav
Returns JSON with depression-risk probability (mean over 1.5 s windows) and label."""
import sys, json, joblib, numpy as np, torch
from transformers import AutoModel
from common import load_raw, normalise, windows
import os
HERE = os.path.dirname(os.path.abspath(__file__))
def predict(path):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    enc = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(dev).eval()
    clf = joblib.load(f"{HERE}/model/wavlm_l6_lr.joblib")
    a, sr = load_raw(os.path.abspath(path)) if os.path.isabs(path) else load_raw(os.path.abspath(path))
    W = np.stack(windows(normalise(a, sr)))
    with torch.no_grad():
        x = torch.from_numpy(W).to(dev); x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)
        e = enc(x, output_hidden_states=True).hidden_states[6].mean(1).cpu().numpy()
    p = float(clf.predict_proba(e)[:, 1].mean())
    return {"depression_risk": round(p, 4), "label": "depressed" if p > 0.5 else "non-depressed", "windows": len(W),
            "note": "screening aid, not a diagnosis; trained on a corpus with known recording-protocol confounds (see update.md)"}
if __name__ == "__main__":
    print(json.dumps(predict(sys.argv[1]), indent=1))
