# %% [meta]
# {"slug": "asd-xmf-nb09-ablations-shortcuts", "title": "ASD-XMF NB09 Ablations Shortcuts",
#  "gpu": true, "internet": true, "datasets": [],
#  "kernels": ["asd-xmf-nb00-data-audit-manifest", "asd-xmf-nb01-face-pipeline", "asd-xmf-nb08-fusion"], "phase": "B"}

# %% [markdown]
# # NB09 - Ablations, missing-modality robustness and shortcut tests (plan 5.4)
# **Ablations of XMF** (same folds/seeds): remove each modality; remove the handcrafted
# behaviour features; no modality dropout; no evidential head (plain BCE); no temporal
# transformer (frame mean); no cross-modal attention; clip-level vs child-level decisions.
#
# **Shortcut tests - a model that only sees these must fail:**
# 1. face-masked background thumbnails -> ResNet-18
# 2. recording metadata (device, room, recorder, site) -> logistic regression
# 3. image-quality statistics (brightness, contrast, blur) -> logistic regression
# 4. face-detection rate only
# If any shortcut's AUROC CI lies above 0.6, the data have a confound to fix **before** publishing.

# %%
import numpy as np, pandas as pd
from pathlib import Path
import asd_models as M
OUT = U.WORK / "nb09"; OUT.mkdir(parents=True, exist_ok=True)
MAN = U.find_path("manifest.csv")
FACE = U.find_path("face_index.csv").parent
BUNDLE = U.find_path("bundle.npz")
man = pd.read_csv(MAN)
FOLDS = [0] if PREFLIGHT else sorted(man.fold_s0[man.fold_s0 >= 0].unique())
SEEDS = (0,) if PREFLIGHT else (0, 1, 2)
EPOCHS = 3 if PREFLIGHT else 150
ALL = ["face", "voice", "cry", "quest"]
ABL = {
    "xmf_minus_face": {"mods": ["voice", "cry", "quest"]}, "xmf_minus_voice": {"mods": ["face", "cry", "quest"]},
    "xmf_minus_cry": {"mods": ["face", "voice", "quest"]}, "xmf_minus_quest": {"mods": ["face", "voice", "cry"]},
    "xmf_no_handcrafted": {"mods": ALL, "no_handcrafted": True}, "xmf_no_moddrop": {"mods": ALL, "modality_dropout": 0.0},
    "xmf_no_evidential": {"mods": ALL, "no_evidential": True}, "xmf_no_temporal": {"mods": ALL, "no_temporal": True},
    "xmf_no_cross": {"mods": ALL, "no_cross": True}, "xmf_clip_level": {"mods": ALL, "clip_level": True},
}
if PREFLIGHT:
    ABL = {k: ABL[k] for k in ("xmf_minus_face", "xmf_no_handcrafted", "xmf_clip_level")}
jobs = [{"name": f"{t}_s{s}_f{f}", "model": "xmf", "model_tag": f"ablation_{t}", "seed": s, "fold": int(f),
         "bundle": str(BUNDLE), "manifest": str(MAN), "out_dir": str(OUT),
         "cfg": {"modality_dropout": 0.5, "max_drop": 2, "epochs": EPOCHS, **cfg}}
        for t, cfg in ABL.items() for s in SEEDS for f in FOLDS]
with U.Timer("ablations"):
    U.run_jobs_parallel("asd_models", "fusion_fold_task", jobs, log_dir=OUT / "logs")

# %% [markdown]
# ## Shortcut 1 - background only (face masked)

# %%
bgs = OUT / "bgstore"
for d in ("crops", "valid"):
    (bgs / d).mkdir(parents=True, exist_ok=True)
idx = pd.read_csv(FACE / "face_index.csv")
for cid in idx.child_id.unique():
    src = FACE / "bg" / f"{cid}.npy"
    if src.exists():
        a = np.load(src, mmap_mode="r")
        dst = bgs / "crops" / f"{cid}.npy"
        if not dst.exists():
            try:
                os.symlink(src, dst)
            except OSError:
                np.save(dst, np.asarray(a))
        np.save(bgs / "valid" / f"{cid}.npy", np.ones(a.shape[:2], bool))
idx.to_csv(bgs / "face_index.csv", index=False)
jobs = [{"name": f"bg_resnet18_s0_f{f}", "model": "resnet18", "model_tag": "SHORTCUT_background_cnn", "seed": 0, "fold": int(f),
         "manifest": str(MAN), "face_root": str(bgs), "out_dir": str(OUT), "img_size": 112, "epochs": 1 if PREFLIGHT else 10,
         "patience": 3, "batch": 16 if PREFLIGHT else 128, "lr": 1e-3, "k_per_clip": 4, "frozen_epochs": 0,
         "max_steps": 3 if PREFLIGHT else None} for f in FOLDS]
U.run_jobs_parallel("asd_models", "frame_fold_task", jobs, log_dir=OUT / "logs")

# %% [markdown]
# ## Shortcuts 2-4 - metadata, image quality, detection rate (logistic regression, same folds)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
qual = []
for cid in man.child_id:
    p = FACE / "qual" / f"{cid}.npy"
    v = FACE / "valid" / f"{cid}.npy"
    if p.exists():
        q = np.load(p).astype(np.float32)
        nz = q.sum(-1) != 0
        qq = q[nz] if nz.any() else q.reshape(-1, q.shape[-1])
        qual.append({"child_id": cid, "bright": qq[:, 0].mean(), "contrast": qq[:, 1].mean(), "blur": qq[:, 2].mean(),
                     "det_rate": float(np.load(v).mean())})
qual = pd.DataFrame(qual)
meta_cols = [c for c in ("device", "room", "recorder", "site") if c in man and man[c].nunique() > 1]
X_meta = pd.get_dummies(man.set_index("child_id")[meta_cols].astype(str)) if meta_cols else None
SHORT = {"SHORTCUT_metadata": X_meta,
         "SHORTCUT_image_quality": qual.set_index("child_id")[["bright", "contrast", "blur"]] if len(qual) else None,
         "SHORTCUT_detection_rate": qual.set_index("child_id")[["det_rate"]] if len(qual) else None}
for name, X in SHORT.items():
    if X is None or X.shape[1] == 0:
        print(name, "- not testable (no variation)")
        continue
    for s in SEEDS:
        for f in FOLDS:
            tr, va, te = U.fold_ids(man, s, f)
            tr, va, te = [[c for c in ids if c in X.index] for ids in (tr, va, te)]
            y = man.set_index("child_id").label
            if len(set(y.loc[tr])) < 2:
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X.loc[tr].astype(float), y.loc[tr])
            for part, ids in (("val", va), ("test", te)):
                if ids:
                    d = pd.DataFrame({"child_id": ids, "prob": clf.predict_proba(X.loc[ids].astype(float))[:, 1], "part": part})
                    d["y_true"] = d.child_id.map(y)
                    U.save_preds(d, name, f, OUT / "preds", s)

# %% [markdown]
# ## Tables

# %%
preds = U.load_all_preds([OUT / "preds"])
rows = []
for m, g in preds.groupby("model"):
    s = U.summarise_model(g, n_boot=200 if PREFLIGHT else 2000)
    ci = s["oof_ci"].set_index("metric").loc["auroc"] if "oof_ci" in s else None
    rows.append({"model": m, "oof_auroc": s.get("oof", {}).get("auroc"),
                 "lo": None if ci is None else ci.lo, "hi": None if ci is None else ci.hi,
                 "test_auroc": s.get("test", {}).get("auroc"),
                 "verdict": ("CONFOUND - fix data" if (m.startswith("SHORTCUT") and ci is not None and ci.lo > 0.6) else
                             ("ok (near chance)" if m.startswith("SHORTCUT") else ""))})
res = pd.DataFrame(rows)
print(U.table_md(res))
res.to_csv(OUT / "ablation_shortcut_table.csv", index=False)
clip = [json.loads(p.read_text()) for p in (OUT / "flags").glob("xmf_clip_level_*.done")]
if clip:
    print("clip-level AUROC (mean over folds):", pd.DataFrame(clip).reindex(columns=["clip_auc_val", "clip_auc_test"]).mean().round(3).to_dict())
if man.is_synthetic.iloc[0]:
    print(U.SYNTHETIC_BANNER)
