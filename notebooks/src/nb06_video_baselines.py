# %% [meta]
# {"slug": "asd-xmf-nb06-video-baselines", "title": "ASD-XMF NB06 Video Baselines",
#  "gpu": true, "internet": true, "datasets": [],
#  "kernels": ["asd-xmf-nb00-data-audit-manifest", "asd-xmf-nb01-face-pipeline", "asd-xmf-nb05-frame-embeddings"],
#  "phase": "B"}

# %% [markdown]
# # NB06 - Temporal / video baselines
# * **Temporal transformer** on cached NB05 embeddings + behaviour features (face branch of
#   XMF alone) - cheap, 5 folds x 3 seeds.
# * **R(2+1)D-18** (Kinetics), **X3D-S** (PyTorchVideo hub), **VideoMAE-S** (HF) fine-tuned
#   end-to-end on 16-frame face clips - expensive, seed 0 only.
# Same locked child-wise folds as every other notebook.

# %%
import numpy as np, pandas as pd
from pathlib import Path
import asd_models as M
OUT = U.WORK / "nb06"; OUT.mkdir(parents=True, exist_ok=True)
MAN = U.find_path("manifest.csv")
FACE = U.find_path("face_index.csv").parent
EMB = U.find_path("emb_efficientnet_b0_expr.npy") or U.find_path("emb_imagenet_effb0_pca.npy")  # fallback if NB03 weights are missing
print("face embedding:", EMB)
man = pd.read_csv(MAN)
FOLDS = [0] if PREFLIGHT else sorted(man.fold_s0[man.fold_s0 >= 0].unique())
SEEDS = (0,) if PREFLIGHT else (0, 1, 2)

# %% [markdown]
# ## 1. Temporal transformer on embeddings

# %%
bundle = OUT / "bundle_face.npz"
M.build_fusion_bundle(MAN, EMB, FACE, None, out_path=bundle, quest_cols=[])
jobs = [{"name": f"tt_s{s}_f{f}", "model": "xmf", "model_tag": "video_temporal_transformer", "seed": s, "fold": int(f),
         "bundle": str(bundle), "manifest": str(MAN), "out_dir": str(OUT),
         "cfg": {"mods": ["face"], "no_evidential": True, "modality_dropout": 0.0, "epochs": 3 if PREFLIGHT else 150}}
        for s in SEEDS for f in FOLDS]
U.run_jobs_parallel("asd_models", "fusion_fold_task", jobs, log_dir=OUT / "logs")

# %% [markdown]
# ## 2. End-to-end video CNN / transformer baselines

# %%
U.pip_install("pytorchvideo")
VIDEO = {"r2plus1d_18": dict(batch=16, lr=3e-4), "x3d_s": dict(batch=16, lr=3e-4), "videomae_s": dict(batch=8, lr=5e-5)}
jobs = [{"name": f"{m}_s0_f{f}", "model": m, "model_tag": f"video_{m}", "seed": 0, "fold": int(f),
         "manifest": str(MAN), "face_root": str(FACE), "out_dir": str(OUT),
         "epochs": 1 if PREFLIGHT else 10, "patience": 3, "max_steps": 2 if PREFLIGHT else None, **kw}
        for m, kw in VIDEO.items() for f in FOLDS]
if PREFLIGHT:
    jobs = [j for j in jobs if j["model"] == "r2plus1d_18"]
codes = U.run_jobs_parallel("asd_models", "video_fold_task", jobs, log_dir=OUT / "logs")
print("exit codes", codes)

# %%
preds = U.load_all_preds([OUT])
rows = []
for m, g in preds.groupby("model"):
    s = U.summarise_model(g, n_boot=200 if PREFLIGHT else 1000)
    rows.append({"model": m, "oof_auroc": s.get("oof", {}).get("auroc"), "test_auroc": s.get("test", {}).get("auroc")})
print(U.table_md(pd.DataFrame(rows)))
if man.is_synthetic.iloc[0]:
    print(U.SYNTHETIC_BANNER)
