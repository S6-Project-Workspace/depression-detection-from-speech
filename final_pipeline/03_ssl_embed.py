"""Stage 3: frozen SSL embeddings (layer-wise mean-pooled) per 1.5 s window.
usage: python 03_ssl_embed.py <hf_model_id> <norm|resamp>"""
import sys, numpy as np, torch
from transformers import AutoModel
from common import *

mid, tag = sys.argv[1], sys.argv[2]
name = mid.split("/")[-1]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModel.from_pretrained(mid).to(dev).eval()
W = np.load(f"{CACHE}/windows_{tag}.npy")
out = []
with torch.no_grad():
    for s in range(0, len(W), 16):
        x = torch.from_numpy(W[s:s + 16]).to(dev)
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)
        hs = model(x, output_hidden_states=True).hidden_states
        out.append(torch.stack([h.mean(1) for h in hs], 1).float().cpu().numpy().astype(np.float16))
        if (s // 16) % 50 == 0: print(name, tag, s, "/", len(W), flush=True)
np.save(f"{CACHE}/emb_{name}_{tag}.npy", np.concatenate(out))
print("done", name, tag)
