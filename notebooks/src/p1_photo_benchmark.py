# %% [meta]
# {"slug": "asd-xmf-p1-photo-benchmark", "title": "ASD-XMF P1 Photo Benchmark",
#  "gpu": true, "internet": true, "datasets": [], "kernels": ["asd-xmf-d1-photo-dataset"], "phase": "A"}

# %% [markdown]
# # P1 - Photo sector: proposed XMF-Face encoder vs 11 baselines (XMF-Photo, 7 expressions)
# **Proposed (plan 5.1):** XMF-Face = EfficientNet-B0 backbone + 256-d expression embedding head (the
# embedding every cohort frame is later mapped to).  **Baselines (plan 5.3 + proposal):** ResNet-18/50,
# EfficientNet-B0/V2-S, MobileNetV3-L, ConvNeXt-T, ViT-S/16, DeiT-S, Swin-T, YOLOv8-cls, YOLOv8-ATT.
#
# Identical protocol for all: same splits, 160 px, max 20 epochs, early stopping on validation macro-F1
# (patience 5), AdamW + cosine, label smoothing 0.1, mixup on half the batches, GPU augmentation, fp16.
# Over/underfitting is reported for every model (train vs validation accuracy gap, learning curves).
# Test sets (FER2013, RAF-DB, external CK+) are used once.

# %%
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import asd_bench as B
U.pip_install("ultralytics")
OUT = U.WORK / "p1"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = U.find_path("images.npy").parent
MODELS = ["xmf_face", "efficientnet_b0", "resnet18", "resnet50", "efficientnetv2_s", "mobilenetv3_l",
          "convnext_tiny", "vit_s16", "deit_s", "swin_t", "yolov8_cls", "yolov8_att"]
HEAVY = {"convnext_tiny", "vit_s16", "deit_s", "swin_t", "efficientnetv2_s", "resnet50"}
if PREFLIGHT:
    MODELS = ["xmf_face", "resnet18"]
jobs = [{"name": m, "model": m, "cache": str(CACHE), "out_dir": str(OUT), "img_size": 160,
         "epochs": 1 if PREFLIGHT else 20, "patience": 5, "batch": 16 if PREFLIGHT else (128 if m in HEAVY else 256),
         "lr": (5e-4 if m in ("vit_s16", "deit_s", "swin_t", "convnext_tiny") else 1.5e-3), "seed": 0, "workers": 2,
         "limit": 600 if PREFLIGHT else None, "max_steps": 4 if PREFLIGHT else None} for m in MODELS]
with U.Timer("photo benchmark"):
    U.run_jobs_parallel("asd_bench", "photo_task", jobs, log_dir=OUT / "logs")

# %% [markdown]
# ## Comparison table (proposed first), with over/underfitting check

# %%
res = {m: json.loads((OUT / "results" / f"{m}.json").read_text()) for m in MODELS if (OUT / "results" / f"{m}.json").exists()}
rows = []
for m, r in res.items():
    row = {"model": m, "proposed": r["proposed"], "params_m": round(r["params_m"], 2), "best_epoch": r["best_epoch"],
           "train_acc": r["train"]["acc"], "val_acc": r["val"]["acc"], "gap": r["gap_train_minus_val_acc"],
           "test_acc": r["test"]["acc"], "test_acc_ci": f"[{r['test_acc_ci'][0]:.3f}, {r['test_acc_ci'][1]:.3f}]",
           "test_macro_f1": r["test"]["macro_f1"], "minutes": round(r["train_minutes"], 1)}
    for s, mm in r["test_by_source"].items():
        row[f"acc_{s}"] = mm.get("acc")
    rows.append(row)
tab = pd.DataFrame(rows)
tab["fit"] = np.where(tab.gap > 0.10, "overfit?", np.where(tab.train_acc < 0.6, "underfit?", "ok"))
tab = tab.sort_values(["proposed", "test_acc"], ascending=[False, False])
print(tab.to_markdown(index=False, floatfmt=".3f"))
tab.to_csv(OUT / "photo_comparison.csv", index=False)

# McNemar: proposed vs every baseline on the same test images
y = np.load(CACHE / "labels.npy"); split = np.load(CACHE / "split.npy", allow_pickle=True)
yt = y[split == "test"][: len(np.load(OUT / "results" / "xmf_face_test_pred.npy"))] if (OUT / "results" / "xmf_face_test_pred.npy").exists() else None
if yt is not None:
    pp = np.load(OUT / "results" / "xmf_face_test_pred.npy")
    mc = []
    for m in res:
        if m == "xmf_face":
            continue
        pb = np.load(OUT / "results" / f"{m}_test_pred.npy")
        mc.append({"baseline": m, **U.mcnemar_test(yt, pp, pb)})
    mc = pd.DataFrame(mc)
    if len(mc):
        mc["p_holm"] = U.holm(mc.p.values)
        print(mc.to_markdown(index=False, floatfmt=".4f"))
        mc.to_csv(OUT / "photo_mcnemar_vs_proposed.csv", index=False)

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for m in res:
    c = pd.read_csv(OUT / "curves" / f"{m}.csv")
    lw = 3 if m == "xmf_face" else 1
    ax[0].plot(c.epoch, c.train_acc, lw=lw, label=m); ax[1].plot(c.epoch, c.val_acc, lw=lw, label=m)
ax[0].set(title="Train accuracy", xlabel="epoch"); ax[1].set(title="Validation accuracy", xlabel="epoch")
ax[1].legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT / "photo_learning_curves.png", dpi=200)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(tab.model, tab.test_acc, color=["tab:red" if p else "tab:gray" for p in tab.proposed])
ax.set(xlabel="test accuracy", title="Photo sector: proposed (red) vs baselines"); plt.tight_layout()
plt.savefig(OUT / "photo_comparison.png", dpi=200)
