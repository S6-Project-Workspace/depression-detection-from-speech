"""Stage 3b: ECAPA-TDNN (VoxCeleb) 192-d utterance embeddings per window (SpeechBrain)."""
import numpy as np, torch
from speechbrain.inference.speaker import EncoderClassifier
from common import *
m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=f"{CACHE}/ecapa_pretrained", run_opts={"device": "cpu"})
W = np.load(f"{CACHE}/windows_norm.npy"); out = []
with torch.no_grad():
    for s in range(0, len(W), 64):
        out.append(m.encode_batch(torch.from_numpy(W[s:s + 64])).squeeze(1).numpy())
        if (s // 64) % 20 == 0: print("ecapa", s, "/", len(W), flush=True)
np.save(f"{CACHE}/emb_ecapa_norm.npy", np.concatenate(out)); print("done ecapa")
