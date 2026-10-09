"""Stage 8: channel-randomisation robustness.
Every window (any class, any language) gets an independently drawn random recording channel:
low-pass cutoff, gain, coloured noise (SNR 5-25 dB), optional 8-bit quantisation / short reverb.
Because the channel is independent of the label, a model that survives it is reading the voice,
not the recording campaign. Embeds the perturbed windows with WavLM-base for evaluation."""
import numpy as np, torch
from scipy.signal import butter, sosfiltfilt, fftconvolve
from transformers import AutoModel
from common import *

def perturb(w, rng):
    a = w.copy()
    fc = rng.choice([3500, 4500, 5500, 6500, 7500])
    a = sosfiltfilt(butter(6, fc, btype="low", fs=SR, output="sos"), a).astype(np.float32)
    if rng.random() < 0.3:                       # short synthetic room response
        t = np.arange(int(0.12 * SR)); ir = rng.standard_normal(len(t)) * np.exp(-t / (SR * rng.uniform(0.02, 0.06))); ir[0] = 1
        a = fftconvolve(a, ir / np.abs(ir).sum(), mode="full")[:len(a)].astype(np.float32)
    if rng.random() < 0.3:                       # crude codec: 8-bit quantisation
        a = (np.round(a * 127) / 127).astype(np.float32)
    snr = rng.uniform(5, 25); sig = np.sqrt(np.mean(a ** 2)) + 1e-8
    n = rng.standard_normal(len(a)); f = np.fft.rfft(n); k = np.maximum(np.arange(len(f)), 1)
    n = np.fft.irfft(f / k ** rng.choice([0.0, 0.5, 1.0]), len(a)); n = n / (n.std() + 1e-9) * sig / (10 ** (snr / 20))
    a = (a + n) * 10 ** (rng.uniform(-6, 6) / 20)
    return a.astype(np.float32)

if __name__ == "__main__":
    W = np.load(f"{CACHE}/windows_norm.npy"); rng = np.random.default_rng(1234)
    P = np.stack([perturb(w, rng) for w in W]); np.save(f"{CACHE}/windows_chanrand.npy", P)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(dev).eval(); out = []
    with torch.no_grad():
        for s in range(0, len(P), 16):
            x = torch.from_numpy(P[s:s + 16]).to(dev); x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)
            hs = model(x, output_hidden_states=True).hidden_states
            out.append(torch.stack([h.mean(1) for h in hs], 1).float().cpu().numpy().astype(np.float16))
            if (s // 16) % 100 == 0: print("chanrand", s, "/", len(P), flush=True)
    np.save(f"{CACHE}/emb_wavlm-base-plus_chanrand.npy", np.concatenate(out)); print("done")
