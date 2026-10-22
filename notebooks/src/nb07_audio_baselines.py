# %% [meta]
# {"slug": "asd-xmf-nb07-audio-baselines", "title": "ASD-XMF NB07 Audio Baselines",
#  "gpu": true, "internet": true, "datasets": [],
#  "kernels": ["asd-xmf-nb00-data-audit-manifest", "asd-xmf-nb02-audio-pipeline"], "phase": "B"}

# %% [markdown]
# # NB07 - Voice and cry baselines for ASD vs non-ASD (child level)
# * logistic regression on child-mean eGeMAPS / wav2vec2 / AST vectors
# * EfficientNet-B0 log-mel CNN per clip (one fold per T4), clip probabilities averaged per child
#
# Children with no usable voice (or no cry) are *excluded from that modality's baseline* and
# counted - single-modality models cannot score them, which is the motivation for XMF.

# %%
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import asd_audio as A
OUT = U.WORK / "nb07"; OUT.mkdir(parents=True, exist_ok=True)
MAN = U.find_path("manifest.csv")
AUD = U.find_path("voice_index.csv").parent
man = pd.read_csv(MAN)
FOLDS = [0] if PREFLIGHT else sorted(man.fold_s0[man.fold_s0 >= 0].unique())
SEEDS = (0,) if PREFLIGHT else (0, 1, 2)

# %%
def child_vectors(tag, name):
    p = AUD / f"{tag}_{name}.npy"
    if not p.exists():
        return None
    X = np.load(p).astype(np.float32)
    if name == "w2v":
        X = X[:, 4:10].mean(1)
    idx = pd.read_csv(AUD / f"{tag}_index.csv")
    if tag == "voice":
        ok = idx.voiced_frac.values >= 0.05
        X, idx = X[ok], idx[ok]
    df = pd.DataFrame(np.nan_to_num(X), index=idx.child_id.values)
    return df.groupby(level=0).mean()

coverage = {}
for tag in ("voice", "cry"):
    for name in ("egemaps", "w2v", "ast"):
        V = child_vectors(tag, name)
        if V is None:
            continue
        coverage[tag] = len(V)
        for s in SEEDS:
            for f in FOLDS:
                tr, va, te = U.fold_ids(man, s, f)
                tr, va, te = [[c for c in ids if c in V.index] for ids in (tr, va, te)]
                ytr = man.set_index("child_id").loc[tr, "label"].values
                if len(set(ytr)) < 2:
                    continue
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")).fit(V.loc[tr], ytr)
                for part, ids in (("val", va), ("test", te)):
                    if ids:
                        d = pd.DataFrame({"child_id": ids, "prob": clf.predict_proba(V.loc[ids])[:, 1], "part": part})
                        d["y_true"] = d.child_id.map(man.set_index("child_id").label)
                        U.save_preds(d, f"{tag}_logreg_{name}", f, OUT / "preds", s)
print("children with modality:", coverage, "of", len(man))

# %% [markdown]
# ## Log-mel CNN per clip

# %%
jobs = []
for tag in ("voice", "cry"):
    ip = AUD / f"{tag}_index.csv"
    if not ip.exists():
        continue
    idx = pd.read_csv(ip).merge(man[["child_id", "label"]], on="child_id")
    if tag == "voice":
        idx = idx[idx.voiced_frac >= 0.05]
    lab_path = OUT / f"{tag}_items_labelled.csv"
    full = pd.read_csv(ip).merge(man[["child_id", "label"]], on="child_id", how="left")
    full.to_csv(lab_path, index=False)
    for f in FOLDS:
        tr, va, te = U.fold_ids(man, 0, f)
        rows = lambda ids: idx.index[idx.child_id.isin(ids)].tolist()
        jobs.append({"name": f"melcnn_{tag}_f{f}", "index": str(lab_path), "mel_dir": str(AUD / f"{tag}_mel"), "label_col": "label",
                     "train_rows": rows(tr), "test_rows": rows(va) + rows(te), "fold": int(f), "out_dir": str(OUT),
                     "seed": 0, "epochs": 1 if PREFLIGHT else 25, "batch": 32, "lr": 1e-3, "max_steps": 3 if PREFLIGHT else None,
                     "crop_frames": 300})
jobs = [j for j in jobs if len(j["train_rows"]) >= 4]
U.run_jobs_parallel("asd_audio", "melcnn_fold_task", jobs, log_dir=OUT / "logs")
for j in jobs:
    p = OUT / "preds" / f"{j['name']}.csv"
    if not p.exists():
        continue
    d = pd.read_csv(p)
    if "p_1" not in d:
        continue
    items = pd.read_csv(j["index"])
    d["child_id"] = items.child_id.values[d.row.values]
    tr, va, te = U.fold_ids(man, 0, j["fold"])
    d["part"] = np.where(d.child_id.isin(va), "val", "test")
    c = d.groupby(["child_id", "part"], as_index=False).agg(prob=("p_1", "mean"), uncertainty=("p_1", "std"))
    c["y_true"] = c.child_id.map(man.set_index("child_id").label)
    U.save_preds(c, f"{j['name'].rsplit('_f', 1)[0]}", j["fold"], OUT / "preds", 0)
    p.unlink()

# %%
preds = U.load_all_preds([OUT / "preds"])
rows = []
for m, g in preds.groupby("model"):
    s = U.summarise_model(g, n_boot=200 if PREFLIGHT else 1000)
    rows.append({"model": m, "children": g.child_id.nunique(), "oof_auroc": s.get("oof", {}).get("auroc"), "test_auroc": s.get("test", {}).get("auroc")})
print(U.table_md(pd.DataFrame(rows)))
if man.is_synthetic.iloc[0]:
    print(U.SYNTHETIC_BANNER)
