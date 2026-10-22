# %% [meta]
# {"slug": "asd-xmf-nb10-xai", "title": "ASD-XMF NB10 XAI",
#  "gpu": true, "internet": true, "datasets": [],
#  "kernels": ["asd-xmf-nb00-data-audit-manifest", "asd-xmf-nb01-face-pipeline", "asd-xmf-nb04-frame-baselines-a-cnn",
#              "asd-xmf-nb08-fusion"], "phase": "B"}

# %% [markdown]
# # NB10 - Explainable AI at five levels, with numbers (plan 6)
# | level | question | method | check |
# |---|---|---|---|
# | modality | did face, voice or cry decide? | exact Shapley over modalities (16 subsets) | sums to P - P(empty) |
# | time | which seconds mattered? | XMF frame attention + Integrated Gradients | rank agreement attention vs IG |
# | behaviour | what did the child do? | SHAP on behaviour features; TCAV concepts (smile, gaze-away) | concept vs random-CAV t-test |
# | pixel | which face regions? | Grad-CAM, Grad-CAM++, Score-CAM, Eigen-CAM | eyes/mouth energy, deletion/insertion AUC, randomisation sanity, stability |
# | case / counterfactual | similar children? what would change it? | kNN in fusion space; single-feature counterfactual | neighbour agreement |
#
# Finally a **blinded clinician-rating packet** (50 explanation cards + rating sheet) for the
# 3-5 clinician study.

# %%
import numpy as np, pandas as pd, torch
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import asd_models as M, asd_xai as X
U.pip_install("grad-cam", "shap")
OUT = U.WORK / "nb10"; OUT.mkdir(parents=True, exist_ok=True)
dev = M.device()
man = pd.read_csv(U.find_path("manifest.csv"))
FACE = U.find_path("face_index.csv").parent
store = M.FaceStore(FACE)
B = M.load_bundle(U.find_path("bundle.npz"))
names = json.loads((FACE / "hand_feature_names.json").read_text())
De, Dh = B["face_dims"]
ckpts = sorted(f for d in U.find_all("ckpt") for f in d.glob("xmf_s0_f*.pt"))
print("XMF fold checkpoints:", [p.name for p in ckpts])
SYN = bool(man.is_synthetic.iloc[0])
pos = {c: i for i, c in enumerate(B["child_id"])}

# %% [markdown]
# ## 1. Modality Shapley, frame attention, IG, fusion embedding (per fold model, its own val children)

# %%
recs, zs = [], []
for ck in ckpts:
    model, info = M.load_fusion_ckpt(ck, dev)
    Bn = M.apply_norm(B, info["norm"])
    tr, va, te = U.fold_ids(man, 0, info["fold"])
    for part, ids in (("val", va), ("test", te), ("train", info["fit_children"])):
        idx = np.array([pos[c] for c in ids if c in pos])
        for s in range(0, len(idx), 32):
            ii = idx[s:s + 32]
            b = M.batch_from_bundle(Bn, ii, dev, model.mods)
            with torch.no_grad():
                o = model(b, return_all=True)
            zs.append(pd.DataFrame(o["z"].cpu().numpy(), index=B["child_id"][ii]).assign(part=part, fold=info["fold"]))
            if part == "train":
                continue
            phi = M.modality_shapley(model, b, model.mods)
            ig = X.integrated_gradients_frames(model, b, steps=8 if PREFLIGHT else 32) if "face" in model.mods else None
            fa = o.get("frame_attn")
            for k, i in enumerate(ii):
                r = {"child_id": B["child_id"][i], "y": int(B["y"][i]), "part": part, "fold": info["fold"],
                     "prob": float(o["prob"][k]), "unc": float(o["unc"][k]),
                     **{f"shap_{m}": float(phi[k, j]) for j, m in enumerate(model.mods)},
                     **{f"gate_{m}": float(o["gates"][k, j]) for j, m in enumerate(model.mods)}}
                if ig is not None and fa is not None:
                    fm = B["face_frame_mask"][i]
                    a, g = fa[k].cpu().numpy()[fm], np.abs(ig[k])[fm]
                    r["attn_vs_ig_spearman"] = float(pd.Series(a).corr(pd.Series(g), method="spearman")) if len(a) > 3 else np.nan
                    r["ig_frames"] = json.dumps(np.round(np.abs(ig[k]) * fm, 5).tolist())
                recs.append(r)
xai = pd.DataFrame(recs)
xai.drop(columns=["ig_frames"], errors="ignore").to_csv(OUT / "xai_per_child.csv", index=False)
mods = [c[5:] for c in xai.columns if c.startswith("shap_")]
glob_imp = xai[[f"shap_{m}" for m in mods]].abs().mean().rename(lambda c: c[5:])
print("mean |Shapley| per modality:\n", glob_imp.round(4).to_string())
if "attn_vs_ig_spearman" in xai:
    print("attention-vs-IG rank agreement:", round(float(xai.attn_vs_ig_spearman.mean()), 3))

# %% [markdown]
# ## 2. Time: which seconds of each stimulus mattered (IG averaged by stimulus)

# %%
fidx = pd.read_csv(FACE / "face_index.csv")
prof = {}
for r in xai.dropna(subset=["ig_frames"]).itertuples():
    ig = np.array(json.loads(r.ig_frames))
    for c, row in fidx[fidx.child_id == r.child_id].iterrows():
        if row.clip_idx < ig.shape[0]:
            prof.setdefault(row.stimulus, []).append(ig[row.clip_idx])
fig, ax = plt.subplots(figsize=(9, 4))
for stim, arr in prof.items():
    a = np.mean(arr, 0)
    ax.plot(np.arange(len(a)) / 5.0, a / (a.max() + 1e-9), label=stim)
ax.set(xlabel="seconds into clip", ylabel="relative |IG|", title="Temporal importance by stimulus")
ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT / "temporal_importance.png", dpi=200); plt.show()

# %% [markdown]
# ## 3. Behaviour: interpretable child-level features + SHAP + counterfactuals

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
KEY = [n for n in names if any(k in n.lower() for k in ("smile", "blink", "browinnerup", "jawopen", "yaw", "pitch", "gaze", "ear_"))]
beh = []
for cid in man.child_id:
    h = store.hand(cid)
    if h is None:
        continue
    st = fidx[fidx.child_id == cid].set_index("clip_idx").stimulus
    row = {"child_id": cid}
    for stim in sorted(fidx.stimulus.unique()):
        cl = [c for c, s_ in st.items() if s_ == stim and c < h.shape[0]]
        if not cl:
            continue
        v = h[cl].reshape(-1, h.shape[-1])
        for n in KEY:
            j = names.index(n)
            val = v[:, j]
            if n in ("yaw", "pitch"):
                val = np.abs(val)
            row[f"{stim}__{n}"] = np.nanmean(val) if np.isfinite(val).any() else np.nan
    beh.append(row)
beh = pd.DataFrame(beh).set_index("child_id")
beh = beh.loc[:, beh.notna().mean() > 0.5]
bcols = beh.columns.tolist()
y = man.set_index("child_id").label
oof = []
for s in (0,) if PREFLIGHT else (0, 1, 2):
    for f in sorted(man[f"fold_s{s}"][man[f"fold_s{s}"] >= 0].unique()):
        tr, va, te = U.fold_ids(man, s, f)
        tr, va, te = [[c for c in ids if c in beh.index] for ids in (tr, va, te)]
        if len(set(y.loc[tr])) < 2:
            continue
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(beh.loc[tr], y.loc[tr])
        for part, ids in (("val", va), ("test", te)):
            if ids:
                d = pd.DataFrame({"child_id": ids, "prob": clf.predict_proba(beh.loc[ids])[:, 1], "part": part, "y_true": y.loc[ids].values})
                U.save_preds(d, "behaviour_logreg", f, OUT / "preds", s)
cv_ids = [c for c in man.child_id[man.split == "cv"] if c in beh.index]
clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(beh.loc[cv_ids], y.loc[cv_ids])
Z = clf[:-1].transform(beh)
try:
    import shap
    sv = shap.LinearExplainer(clf[-1], clf[:-1].transform(beh.loc[cv_ids])).shap_values(Z)
    imp = pd.DataFrame({"feature": bcols, "mean_abs_shap": np.abs(sv).mean(0), "coef": clf[-1].coef_[0]}).sort_values("mean_abs_shap", ascending=False)
    print(U.table_md(imp.head(15)))
    imp.to_csv(OUT / "behaviour_shap.csv", index=False)
except Exception as e:
    print("SHAP failed:", e)
thr = 0.5
cfs = []
for i, cid in enumerate(beh.index):
    if cid in set(man.child_id[man.split == "test"]):
        feat, delta = X.single_feature_counterfactual(clf[-1], Z[i], bcols, thr)
        if feat:
            cfs.append({"child_id": cid, "feature": feat, "delta_sd": round(float(delta), 2)})
cfs = pd.DataFrame(cfs)
print("counterfactuals found for", len(cfs), "test children; most common features:")
print(cfs.feature.value_counts().head(8).to_string() if len(cfs) else "-")
cfs.to_csv(OUT / "counterfactuals.csv", index=False)

# %% [markdown]
# ## 4. TCAV-style concepts in the embedding space (smile, gaze-away) vs random directions

# %%
from scipy import stats
def frames_where(pred, k=3000):
    E, rows = [], 0
    for i, cid in enumerate(B["child_id"]):
        h = store.hand(cid)
        if h is None:
            continue
        C = min(h.shape[0], B["face"].shape[1])
        m = pred(h[:C]) & B["face_frame_mask"][i, :C]
        if m.any():
            E.append(B["face"][i, :C][m][:, :De].astype(np.float32))
    E = np.concatenate(E) if E else np.zeros((0, De), np.float32)
    return E[np.random.default_rng(0).choice(len(E), min(k, len(E)), replace=False)] if len(E) else E
smile_j = [names.index(n) for n in names if "smile" in n.lower()]
yaw_j = names.index("yaw") if "yaw" in names else None
concepts = {}
if smile_j:
    concepts["social_smile"] = (frames_where(lambda h: np.nanmean(h[..., smile_j], -1) > 0.4), frames_where(lambda h: np.nanmean(h[..., smile_j], -1) < 0.1))
if yaw_j is not None:
    concepts["gaze_away"] = (frames_where(lambda h: np.abs(h[..., yaw_j]) > 20), frames_where(lambda h: np.abs(h[..., yaw_j]) < 8))
tcav_rows = []
if ckpts and concepts:
    model, info = M.load_fusion_ckpt(ckpts[0], dev)
    mu, sd = info["norm"]["face"]
    Bn = M.apply_norm(B, info["norm"])
    te = np.array([pos[c] for c in man.child_id[man.split == "test"] if c in pos])
    b = M.batch_from_bundle(Bn, te, dev, model.mods)
    rng = np.random.default_rng(0)
    for cname, (Ep, En) in concepts.items():
        if len(Ep) < 20 or len(En) < 20:
            print(cname, "too few frames"); continue
        z = lambda E: (E - mu[:De]) / sd[:De]
        s_con = X.tcav_score(model, b, X.concept_vector(z(Ep), z(En)), De)
        rand = [(X.tcav_score(model, b, X.concept_vector(z(En[rng.permutation(len(En))[:len(En)//2]]), z(En[rng.permutation(len(En))[:len(En)//2]]), seed=k), De) > 0).mean() for k in range(10)]
        score = float((s_con > 0).mean())
        t = stats.ttest_1samp(rand, score)
        tcav_rows.append({"concept": cname, "tcav_score": score, "random_mean": float(np.mean(rand)), "p_vs_random": float(t.pvalue)})
tcav = pd.DataFrame(tcav_rows)
print(U.table_md(tcav) if len(tcav) else "TCAV skipped")
tcav.to_csv(OUT / "tcav.csv", index=False)

# %% [markdown]
# ## 5. Pixel level: CAM family on the expression-pretrained frame model (NB04, seed 0)

# %%
import timm
frame_ck = sorted(f for d in U.find_all("ckpt") for f in d.glob("efficientnet_b0_expr_s0_f*.pt"))
pix_rows, sanity_rows, stab_rows = [], [], []
if frame_ck:
    fm_ = timm.create_model(M.TIMM_ZOO["efficientnet_b0"], pretrained=False, num_classes=1)
    fm_.load_state_dict(torch.load(frame_ck[0], map_location="cpu"))
    fm_ = fm_.to(dev).eval()
    fold0 = int(frame_ck[0].stem.split("_f")[-1])
    _, va, _ = U.fold_ids(man, 0, fold0)
    imgs, bxs = [], []
    pool_ids = [c for c in va if store.has(c)] or [c for c in man.child_id if store.has(c)]  # fold-0 val children first
    for cid in pool_ids[: 4 if PREFLIGHT else 32]:
        if not store.has(cid):
            continue
        v = store.valid(cid); cr = store.crops(cid)
        bx = np.load(FACE / "boxes" / f"{cid}.npy").astype(np.float32)
        for c in range(min(2, v.shape[0])):
            ts = np.flatnonzero(v[c])
            if len(ts):
                t = ts[len(ts) // 2]; imgs.append(np.asarray(cr[c, t])); bxs.append(bx[c, t])
if frame_ck and imgs:
    x = M.gpu_prepare(torch.from_numpy(np.stack(imgs)).to(dev), 160, train=False).float().contiguous()
    bxs = np.stack(bxs)
    for meth in ("gradcam", "gradcam++", "eigencam") + (() if PREFLIGHT else ("scorecam",)):
        cams = X.compute_cams(fm_, x, meth)
        pix_rows.append({"method": meth, **X.region_energy(cams, bxs), **X.deletion_insertion(fm_, x, cams, steps=10 if PREFLIGHT else 20)})
    rnd = np.random.default_rng(0).random((x.shape[0], 160, 160)).astype(np.float32)
    pix_rows.append({"method": "random_map", **X.region_energy(rnd, bxs), **X.deletion_insertion(fm_, x, rnd, steps=10 if PREFLIGHT else 20)})
    sanity_rows = X.sanity_cascade(fm_, x, "gradcam")
    stab_rows = [X.stability(fm_, x, "gradcam")]
    cams = X.compute_cams(fm_, x[:8], "gradcam")
    fig, axs = plt.subplots(2, min(8, len(cams)), figsize=(2 * min(8, len(cams)), 4))
    for k in range(min(8, len(cams))):
        axs[0, k].imshow(imgs[k]); axs[1, k].imshow(imgs[k]); axs[1, k].imshow(cams[k], alpha=0.5, cmap="jet")
        axs[0, k].axis("off"); axs[1, k].axis("off")
    plt.tight_layout(); plt.savefig(OUT / "gradcam_examples.png", dpi=150); plt.show()
pix = pd.DataFrame(pix_rows); print(U.table_md(pix) if len(pix) else "no frame checkpoint")
print("sanity (should fall towards 0):", sanity_rows); print("stability (should stay high):", stab_rows)
pix.to_csv(OUT / "pixel_xai_metrics.csv", index=False)
U.save_json({"sanity": sanity_rows, "stability": stab_rows}, OUT / "pixel_xai_sanity.json")

# %% [markdown]
# ## 6. Case-based: nearest training children in the fusion embedding

# %%
Zs = pd.concat(zs) if zs else pd.DataFrame()
knn = []
if len(Zs):
    from sklearn.neighbors import NearestNeighbors
    for f, g in Zs.groupby("fold"):
        trn, qry = g[g.part == "train"], g[g.part == "test"]
        if len(trn) < 6 or not len(qry):
            continue
        nn_ = NearestNeighbors(n_neighbors=5, metric="cosine").fit(trn.drop(columns=["part", "fold"]).values)
        _, ind = nn_.kneighbors(qry.drop(columns=["part", "fold"]).values)
        for cid, nb in zip(qry.index, ind):
            labs = y.loc[trn.index[nb]].values
            knn.append({"child_id": cid, "fold": f, "neighbour_asd_frac": labs.mean(), "y": y.loc[cid]})
knn = pd.DataFrame(knn)
if len(knn):
    print("kNN neighbour-label AUROC (case-based plausibility):", round(U._safe_auc(knn.y, knn.neighbour_asd_frac), 3))
knn.to_csv(OUT / "case_based_knn.csv", index=False)

# %% [markdown]
# ## 7. Blinded clinician-rating packet (50 cards)

# %%
cards = OUT / "clinician_packet"; cards.mkdir(exist_ok=True)
sel = xai[xai.part == "test"].drop_duplicates("child_id").sample(frac=1, random_state=1).head(50)
sheet = []
for k, r in enumerate(sel.itertuples()):
    card_id = f"CARD{k+1:03d}"
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].barh(mods, [getattr(r, f"shap_{m}") for m in mods]); ax[0].set_title("contribution to risk")
    if isinstance(getattr(r, "ig_frames", None), str):
        ig = np.array(json.loads(r.ig_frames)); ax[1].imshow(ig, aspect="auto", cmap="magma"); ax[1].set(title="when it mattered", xlabel="frame (5 fps)", ylabel="clip")
    fig.suptitle(f"{card_id}  risk={r.prob:.2f}  uncertainty={r.unc:.2f}"); plt.tight_layout()
    plt.savefig(cards / f"{card_id}.png", dpi=110); plt.close()
    sheet.append({"card_id": card_id, "usefulness_1to5": "", "agreement_1to5": "", "comment": ""})
pd.DataFrame(sheet).to_csv(cards / "rating_sheet_BLINDED.csv", index=False)
sel.assign(card_id=[s["card_id"] for s in sheet])[["card_id", "child_id", "y", "prob"]].to_csv(OUT / "rating_key_DO_NOT_SHARE.csv", index=False)
print("cards:", len(sheet), "->", cards)
if SYN:
    print(U.SYNTHETIC_BANNER)
