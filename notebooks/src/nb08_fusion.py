# %% [meta]
# {"slug": "asd-xmf-nb08-fusion", "title": "ASD-XMF NB08 Fusion",
#  "gpu": true, "internet": true, "datasets": [],
#  "kernels": ["asd-xmf-nb00-data-audit-manifest", "asd-xmf-nb01-face-pipeline", "asd-xmf-nb02-audio-pipeline",
#              "asd-xmf-nb05-frame-embeddings"], "phase": "B"}

# %% [markdown]
# # NB08 - BanglaASD-XMF and six fusion baselines
# **Proposed (XMF).** Face-video branch (2-layer temporal transformer over NB05 expression
# embeddings + 64 behaviour features per frame), voice, cry and M-CHAT branches -> learned
# *missing* tokens -> bidirectional cross-modal attention (face <-> voice/cry) -> clip-to-child
# attention pooling -> gated fusion -> evidential head (P(ASD) + uncertainty -> "refer").
# Trained with modality dropout (1-2 modalities hidden per child) so it works for preverbal
# children and partial recordings; temperature-scaled on an inner split.
#
# **Baselines** (same bundle, folds and seeds): early concatenation, late averaging,
# stacking, Tensor Fusion Network, Low-rank Multimodal Fusion, MulT-style crossmodal
# transformer.  Every model is run **with and without the questionnaire branch**, because
# M-CHAT overlaps with the screening reference (plan 5.1).
#
# 5 folds x 3 seeds per model; two jobs at a time on the two T4s.  Fold models of XMF are
# saved for NB10 (XAI) and NB12 (export).

# %%
import numpy as np, pandas as pd
from pathlib import Path
import asd_models as M
OUT = U.WORK / "nb08"; (OUT / "ckpt").mkdir(parents=True, exist_ok=True)
MAN = U.find_path("manifest.csv")
FACE = U.find_path("face_index.csv").parent
EMB = U.find_path("emb_efficientnet_b0_expr.npy") or U.find_path("emb_imagenet_effb0_pca.npy")  # fallback if NB03 weights are missing
print("face embedding:", EMB)
AUD = U.find_path("voice_fusion_index.csv")
AUD = AUD.parent if AUD else None
man = pd.read_csv(MAN)
FOLDS = [0] if PREFLIGHT else sorted(man.fold_s0[man.fold_s0 >= 0].unique())
SEEDS = (0,) if PREFLIGHT else (0, 1, 2)
EPOCHS = 3 if PREFLIGHT else 150
bundle = OUT / "bundle.npz"
B = M.build_fusion_bundle(MAN, EMB, FACE, AUD, out_path=bundle)
print({k: v.shape for k, v in B.items() if isinstance(v, np.ndarray)})
print("modality availability (children):", {m: float(B[f"{m}_mask"].reshape(len(B["y"]), -1).any(1).mean()) for m in M.MODALITIES if m in B})

# %%
ALL = ["face", "voice", "cry", "quest"]
NOQ = ["face", "voice", "cry"]
MODELS = {
    "xmf": ("xmf", {"mods": ALL}), "xmf_noquest": ("xmf", {"mods": NOQ}),
    "early_concat": ("early_concat", {"mods": ALL}), "early_concat_noquest": ("early_concat", {"mods": NOQ}),
    "late_avg": ("late_avg", {"mods": ALL}), "late_avg_noquest": ("late_avg", {"mods": NOQ}),
    "stacking": ("stacking", {"mods": ALL}), "stacking_noquest": ("stacking", {"mods": NOQ}),
    "tfn": ("tfn", {"mods": ALL}), "tfn_noquest": ("tfn", {"mods": NOQ}),
    "lmf": ("lmf", {"mods": ALL}), "lmf_noquest": ("lmf", {"mods": NOQ}),
    "mult": ("mult", {"mods": ALL}), "mult_noquest": ("mult", {"mods": NOQ}),
}
if PREFLIGHT:
    MODELS = {k: v for k, v in MODELS.items() if k in ("xmf", "xmf_noquest", "early_concat", "stacking", "tfn", "mult")}
jobs = []
for tag, (arch, cfg) in MODELS.items():
    for s in SEEDS:
        for f in FOLDS:
            jobs.append({"name": f"{tag}_s{s}_f{f}", "model": arch, "model_tag": f"fusion_{tag}", "seed": s, "fold": int(f),
                         "bundle": str(bundle), "manifest": str(MAN), "out_dir": str(OUT),
                         "cfg": {**cfg, "epochs": EPOCHS, "modality_dropout": 0.5, "max_drop": 2},
                         "missing_eval": tag.startswith("xmf"), "save_model": tag.startswith("xmf")})
with U.Timer("fusion"):
    codes = U.run_jobs_parallel("asd_models", "fusion_fold_task", jobs, log_dir=OUT / "logs")
print("non-zero exits:", sum(c != 0 for c in codes))

# %% [markdown]
# ## Quick look (full statistics in NB11)

# %%
preds = U.load_all_preds([OUT / "preds"])
rows = []
for m, g in preds.groupby("model"):
    s = U.summarise_model(g, n_boot=200 if PREFLIGHT else 1000)
    t = s.get("test", {})
    rows.append({"model": m, "oof_auroc": s.get("oof", {}).get("auroc"), "test_auroc": t.get("auroc"),
                 "test_sens": t.get("sens"), "test_spec": t.get("spec"), "test_ece": t.get("ece")})
res = pd.DataFrame(rows).sort_values("oof_auroc", ascending=False)
print(U.table_md(res))
res.to_csv(OUT / "fusion_summary.csv", index=False)
rob = [json.loads(p.read_text()).get("missing_auc", {}) for p in (OUT / "flags").glob("xmf_s*_f*.done")]
if rob:
    print("XMF test AUROC with modalities removed at test time (mean over folds):")
    print(pd.DataFrame(rob).mean().round(3).to_string())
if man.is_synthetic.iloc[0]:
    print(U.SYNTHETIC_BANNER)
