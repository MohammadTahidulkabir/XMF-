# %% [meta]
# {"slug": "asd-xmf-nb04-frame-baselines", "title": "ASD-XMF NB04 Frame Baselines",
#  "gpu": true, "internet": true, "datasets": [],
#  "kernels": ["asd-xmf-nb00-data-audit-manifest", "asd-xmf-nb01-face-pipeline", "asd-xmf-p1-photo-benchmark"],
#  "phase": "B",
#  "variants": [
#   {"suffix": "a-cnn", "title": "ASD-XMF NB04 Frame Baselines A CNN", "vars": {"MODELS": ["resnet18", "resnet50", "efficientnet_b0", "efficientnet_b0_expr"]}},
#   {"suffix": "b-cnn", "title": "ASD-XMF NB04 Frame Baselines B CNN", "vars": {"MODELS": ["efficientnetv2_s", "mobilenetv3_l", "mobilenetv3_l_expr", "convnext_tiny"]}},
#   {"suffix": "c-vit", "title": "ASD-XMF NB04 Frame Baselines C VIT", "vars": {"MODELS": ["vit_s16", "deit_s", "swin_t"]}},
#   {"suffix": "d-yolo", "title": "ASD-XMF NB04 Frame Baselines D YOLO", "vars": {"MODELS": ["yolov8_cls", "yolov8_att"]}}
#  ]}

# %% [markdown]
# # NB04 - Frame-level baselines (one model family per notebook)
# Static per-frame classifiers on the aligned face crops: the comparison models of plan 5.3.
# Every model uses **the same** locked child-wise folds, 160 px input, epoch budget, early
# stopping on validation AUROC (child level) and seeds.  A child's score is the mean over its
# clips of the mean frame probability.
#
# Speed: crops are memory-mapped uint8, augmentation runs on the GPU, fp16 autocast,
# channels_last, linear-probe epoch then full fine-tune, and two folds train at once (one per T4).
# `*_expr` models start from the NB03 expression weights; the rest from ImageNet.

# %%
import numpy as np, pandas as pd
from pathlib import Path
if any(m.startswith("yolov8") for m in VARIANT["MODELS"]):
    U.pip_install("ultralytics")
OUT = U.WORK / "nb04"; OUT.mkdir(parents=True, exist_ok=True)
MAN = U.find_path("manifest.csv")
FACE = U.find_path("face_index.csv").parent
man = pd.read_csv(MAN)
SEEDS = (0,) if PREFLIGHT else (0,)      # frame baselines: seed 0; set (0, 1, 2) if quota allows
FOLDS = [0] if PREFLIGHT else sorted(man.fold_s0[man.fold_s0 >= 0].unique())
print("models:", VARIANT["MODELS"], "| folds:", FOLDS, "| synthetic:", bool(man.is_synthetic.iloc[0]))

def init_for(model):
    if not model.endswith("_expr"):
        return None
    p = U.find_path(f"{model[:-5]}_expr_backbone.pt")
    print(model, "init:", p)
    return str(p) if p else None

jobs = []
for model in VARIANT["MODELS"]:
    for s in SEEDS:
        for f in FOLDS:
            jobs.append({"name": f"{model}_s{s}_f{f}", "model": model, "model_tag": f"frame_{model}", "seed": s, "fold": int(f),
                         "manifest": str(MAN), "face_root": str(FACE), "out_dir": str(OUT), "init_weights": init_for(model),
                         "img_size": 160, "epochs": 1 if PREFLIGHT else 12, "patience": 4, "frozen_epochs": 1,
                         "batch": 16 if PREFLIGHT else (96 if model in ("vit_s16", "deit_s", "swin_t", "convnext_tiny") else 128),
                         "lr": 1e-3, "backbone_mult": 0.1, "k_per_clip": 8, "workers": 2,
                         "max_steps": 4 if PREFLIGHT else None,
                         "keep_ckpt": (model == "efficientnet_b0_expr" and s == 0)})   # NB10 Grad-CAM uses these
with U.Timer("frame baselines"):
    codes = U.run_jobs_parallel("asd_models", "frame_fold_task", jobs, log_dir=OUT / "logs")
print("exit codes:", codes)

# %%
preds = U.load_all_preds([OUT])
rows = []
for m, g in preds.groupby("model"):
    s = U.summarise_model(g, n_boot=200 if PREFLIGHT else 1000)
    rows.append({"model": m, "oof_auroc": s.get("oof", {}).get("auroc"), "test_auroc": s.get("test", {}).get("auroc"),
                 "test_sens": s.get("test", {}).get("sens"), "test_spec": s.get("test", {}).get("spec")})
res = pd.DataFrame(rows)
print(U.table_md(res))
res.to_csv(OUT / "frame_baselines_summary.csv", index=False)
if man.is_synthetic.iloc[0]:
    print(U.SYNTHETIC_BANNER)
