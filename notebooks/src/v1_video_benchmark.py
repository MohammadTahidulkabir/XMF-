# %% [meta]
# {"slug": "asd-xmf-v1-video-benchmark", "title": "ASD-XMF V1 Video Benchmark",
#  "gpu": true, "internet": true, "datasets": [], "kernels": ["asd-xmf-d3-video-dataset"], "phase": "A"}

# %% [markdown]
# # V1 - Video sector: proposed XMF temporal encoder vs baselines (MMASD behaviour activities)
# **Proposed (plan 5.1):** XMF-Temporal = per-frame projection -> 2-layer transformer -> attention pooling
# (the same temporal encoder the face-video branch uses).  **Baselines:** pooled-statistics MLP, 1D-CNN,
# BiLSTM, CNN-LSTM (proposal).  5-fold StratifiedGroupKFold **by child** (no child in both train and test),
# an inner child-wise validation split for early stopping, identical folds for every model.

# %%
import numpy as np, pandas as pd
import asd_bench as B
SRC = U.find_path("skeleton.npy").parent
OUT = U.WORK / "v1"; OUT.mkdir(parents=True, exist_ok=True)
X = np.load(SRC / "skeleton.npy"); Mk = np.load(SRC / "mask.npy")
meta = pd.read_csv(SRC / "manifest.csv", dtype={"subject": str})
classes = sorted(meta.activity.unique()); y = meta.activity.map({c: i for i, c in enumerate(classes)}).values
print(X.shape, len(classes), "activities,", meta.subject.nunique(), "children")
with U.Timer("video benchmark"):
    rows = B.seq_cv(X, Mk, y, meta.subject.values, len(classes), list(B.SEQ_MODELS), OUT, "mmasd_activity",
                    epochs=3 if PREFLIGHT else 150, folds=3 if PREFLIGHT else 5, max_steps=5 if PREFLIGHT else None)
tab = B.comparison_table(rows)
tab["majority_baseline"] = pd.Series(y).value_counts(normalize=True).iloc[0]
print(tab.to_markdown(index=False, floatfmt=".3f"))
tab.to_csv(OUT / "video_comparison.csv", index=False)
