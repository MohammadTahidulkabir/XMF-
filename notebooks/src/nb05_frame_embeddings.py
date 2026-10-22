# %% [meta]
# {"slug": "asd-xmf-nb05-frame-embeddings", "title": "ASD-XMF NB05 Frame Embeddings",
#  "gpu": true, "internet": true, "datasets": [],
#  "kernels": ["asd-xmf-nb00-data-audit-manifest", "asd-xmf-nb01-face-pipeline", "asd-xmf-p1-photo-benchmark"],
#  "phase": "B"}

# %% [markdown]
# # NB05 - Per-frame 256-d embedding cache
# Every valid frame of every clip is embedded **once** with backbones that never saw the
# cohort labels (NB03 expression nets trained on public data only), so the fusion folds in
# NB08 cannot leak through the embedding.  An ImageNet EfficientNet-B0 (PCA -> 256, fitted
# on CV children only) is cached as a control.
#
# Output: `emb/emb_<backbone>.npy` float16 `[n_clips, 40, 256]` + `_index.csv`.

# %%
import numpy as np, pandas as pd, torch
from pathlib import Path
import asd_models as M
OUT = U.WORK / "emb"; OUT.mkdir(parents=True, exist_ok=True)
man = pd.read_csv(U.find_path("manifest.csv"))
store = M.FaceStore(U.find_path("face_index.csv").parent)
ids = [c for c in man.child_id if store.has(c)]
if PREFLIGHT:
    ids = ids[:6]
dev = M.device()
CKPT = U.find_path("efficientnet_b0_expr_backbone.pt")
CKPT = CKPT.parent if CKPT else None
print("children:", len(ids), "| NB03 ckpt dir:", CKPT)

# %%
for base in ("efficientnet_b0", "mobilenetv3_l"):
    net = M.load_expr_net(base, CKPT) if CKPT else None
    if net is None:
        print("missing NB03 weights for", base, "- skipped")
        continue
    net = net.to(dev).to(memory_format=torch.channels_last)
    with U.Timer(f"embed {base}_expr"):
        M.embed_store(net, store, ids, 160, OUT / f"emb_{base}_expr.npy", 256, batch=256, fwd=net.embed)

# %%
import timm
from sklearn.decomposition import PCA
net = timm.create_model(M.TIMM_ZOO["efficientnet_b0"], pretrained=True, num_classes=0).to(dev).eval()
tmp = OUT / "_imnet_raw.npy"
with U.Timer("embed imagenet effb0"):
    E = M.embed_store(net, store, ids, 160, tmp, net.num_features, batch=256)
idx = pd.read_csv(str(tmp).replace(".npy", "_index.csv"))
cv_ids = set(man.child_id[man.split == "cv"])
rows = np.flatnonzero(idx.child_id.isin(cv_ids).values)
sample = E[rows].reshape(-1, E.shape[-1]).astype(np.float32)
sample = sample[np.abs(sample).sum(1) > 0]
sample = sample[np.random.default_rng(0).choice(len(sample), min(len(sample), 50000), replace=False)]
pca = PCA(256 if sample.shape[0] > 256 else min(sample.shape) - 1, random_state=0).fit(sample)
Z = np.lib.format.open_memmap(OUT / "emb_imagenet_effb0_pca.npy", mode="w+", dtype=np.float16, shape=E.shape[:2] + (pca.n_components_,))
for i in range(len(E)):
    x = E[i].astype(np.float32)
    nz = np.abs(x).sum(1) > 0
    z = np.zeros((x.shape[0], pca.n_components_), np.float32)
    if nz.any():
        z[nz] = pca.transform(x[nz])
    Z[i] = z.astype(np.float16)
Z.flush()
idx.to_csv(OUT / "emb_imagenet_effb0_pca_index.csv", index=False)
del E; tmp.unlink(); Path(str(tmp).replace(".npy", "_index.csv")).unlink()
print("PCA explained variance (256):", round(float(pca.explained_variance_ratio_.sum()), 3))
for f in sorted(OUT.glob("*.npy")):
    print(f.name, np.load(f, mmap_mode="r").shape, round(f.stat().st_size / 1e6, 1), "MB")
