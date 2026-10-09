"""Stage 7: figures + markdown tables from the JSON results."""
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RES

audit = json.load(open(f"{RES}/audit.json")); cf = json.load(open(f"{RES}/checkF.json"))
ev = json.load(open(f"{RES}/eval_results.json")); ls = json.load(open(f"{RES}/eval_layerscan.json"))

# Fig 1: Check F
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for a, lang in zip(ax, ["Malayalam", "Tamil"]):
    v = ["raw16k", "norm_v1", "norm_v2_nfe"]; x = np.arange(3)
    a.bar(x - .2, [cf[f"{t}|{lang}|noise_floor_only_AUC"] for t in v], .4, label="non-speech frames only", color="#c0392b")
    a.bar(x + .2, [cf[f"{t}|{lang}|speech_frames_only_AUC"] for t in v], .4, label="speech frames only", color="#2980b9")
    a.axhline(.5, ls="--", c="k", lw=.8); a.set_xticks(x, ["resample only", "+band-limit/window", "+noise-floor eq."]); a.set_title(lang); a.set_ylim(.4, 1.02)
ax[0].set_ylabel("source-held-out AUC"); ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{RES}/fig_checkF.png", dpi=150); plt.close()

# Fig 2: layer scan
plt.figure(figsize=(6, 3.6))
for nm, sc in ls["layer_scan_malayalam_LOSO_LR"].items(): plt.plot(sc, marker="o", ms=3, label=nm)
plt.xlabel("transformer layer"); plt.ylabel("Malayalam LOSO AUC (LR probe)"); plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{RES}/fig_layerscan.png", dpi=150); plt.close()

# Table
rows = ["| Features | Head | k | Mal LOSO bal-acc | Mal LOSO AUC | Mal TEST bal-acc | Mal TEST 48k-slice bal-acc | Tam→Mal AUC | Mal→Tam AUC |", "|---|---|---|---|---|---|---|---|---|"]
for r in ev["table"]:
    s = r["TEST_mal"]["slice_48k"]
    rows.append(f"| {r['features']} | {r['model']} | {r['nuisance_k']} | {r['LOSO_mal']['bal_acc']:.3f} | {r['LOSO_mal']['auc']:.3f} | {r['TEST_mal']['bal_acc']:.3f} | {s['bal_acc']:.3f} | {r['XLING_tam2mal']['auc']:.3f} | {r['XLING_mal2tam']['auc']:.3f} |" if s else "")
open(f"{RES}/results_table.md", "w").write("\n".join(rows)); print("\n".join(rows))
