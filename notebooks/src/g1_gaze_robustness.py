# %% [meta]
# {"slug": "asd-xmf-g1-gaze-robustness", "title": "ASD-XMF G1 Gaze Robustness",
#  "gpu": false, "internet": true, "datasets": [], "kernels": [], "phase": "R"}

# %% [markdown]
# # G1 - Is the gaze signal more than recording quality?
# In NB13 the tracking ratio alone (a data-quality measure) reached AUROC 0.84, close to the gaze models.
# A reviewer will ask whether the gaze models only re-learn quality.  Five checks on the same child-level
# features (one row per child, repeated stratified 5-fold CV over children, 10 repeats):
#
# 1. quality only vs gaze only vs gaze + quality (DeLong: does gaze add to quality?)
# 2. gaze features residualised on tracking ratio inside each training fold (quality removed)
# 3. high-quality children only (tracking ratio >= median), where quality varies little
# 4. label-permutation test of the gaze model (200 permutations)
# 5. how strongly each gaze feature correlates with tracking ratio

# %%
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
OUT = U.WORK / "g1"; OUT.mkdir(parents=True, exist_ok=True)
_own = U.WORK / "nb13" / "gaze_child_features.csv"   # written by section nb13 of the same notebook
data = pd.read_csv(_own if _own.exists() else U.find_path("gaze_child_features.csv"))
REPEATS = 3 if PREFLIGHT else 10
N_PERM = 20 if PREFLIGHT else 200
N_BOOT = 300 if PREFLIGHT else 2000
Q = "all_track_ratio"
skip = {"pid", "sex", "age", "cls", "cars", "label", "child_id", "split", "sex_m", Q}
GAZE = [c for c in data.columns if c not in skip and not c.startswith("fold_") and pd.api.types.is_numeric_dtype(data[c])]
y = data.label.values.astype(int)
print(f"{len(data)} children ({y.sum()} ASD) | {len(GAZE)} gaze features | quality = {Q}")

def lr(C):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=5000))

def residualise(tr_df, te_df, cols):
    """Remove the linear effect of tracking ratio from every gaze feature, fitted on the training fold only."""
    q_tr = tr_df[[Q]].fillna(tr_df[Q].median()).values; q_te = te_df[[Q]].fillna(tr_df[Q].median()).values
    a, b = tr_df[cols].copy(), te_df[cols].copy()
    for c in cols:
        ok = a[c].notna().values
        if ok.sum() < 5:
            continue
        reg = LinearRegression().fit(q_tr[ok], a[c].values[ok])
        a[c] = a[c] - reg.predict(q_tr); b[c] = b[c] - reg.predict(q_te)
    return a, b

def cv_prob(df, cols, C=0.1, resid=False, repeats=REPEATS, labels=None):
    """Child-level probability averaged over `repeats` repetitions of stratified 5-fold CV."""
    lab = df.label.values.astype(int) if labels is None else labels
    P = np.zeros(len(df))
    for r in range(repeats):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(df, lab):
            a, b = (residualise(df.iloc[tr], df.iloc[te], cols) if resid else (df.iloc[tr][cols], df.iloc[te][cols]))
            P[te] += lr(C).fit(a, lab[tr]).predict_proba(b)[:, 1] / repeats
    return P

# %% [markdown]
# ## 1-2. Quality, gaze, gaze + quality, gaze with quality removed

# %%
probs = {
    "quality_only": cv_prob(data, [Q], C=1.0),
    "gaze_only": cv_prob(data, GAZE),
    "gaze_plus_quality": cv_prob(data, GAZE + [Q]),
    "gaze_quality_residualised": cv_prob(data, GAZE, resid=True),
}
rows = []
for name, p in probs.items():
    ci = U.bootstrap_ci(y, p, n_boot=N_BOOT)
    rows.append({"analysis": "all children", "model": name, "n": len(y), "auroc": U.fmt_ci(ci)})
tests = []
for a, b in [("gaze_plus_quality", "quality_only"), ("gaze_only", "quality_only"),
             ("gaze_quality_residualised", "quality_only"), ("gaze_quality_residualised", "gaze_only")]:
    tests.append({"model": a, "vs": b, **U.delong_test(y, probs[a], probs[b])})

# %% [markdown]
# ## 3. High-quality children only

# %%
hq = data[data[Q] >= data[Q].median()].reset_index(drop=True)
yh = hq.label.values.astype(int)
print(f"high-quality subset: {len(hq)} children ({yh.sum()} ASD), tracking ratio >= {data[Q].median():.1f}")
if min(yh.sum(), len(yh) - yh.sum()) >= 5:
    ph = {"quality_only": cv_prob(hq, [Q], C=1.0), "gaze_only": cv_prob(hq, GAZE)}
    for name, p in ph.items():
        rows.append({"analysis": "tracking ratio >= median", "model": name, "n": len(yh),
                     "auroc": U.fmt_ci(U.bootstrap_ci(yh, p, n_boot=N_BOOT))})
    tests.append({"model": "gaze_only (high-quality)", "vs": "quality_only (high-quality)",
                  **U.delong_test(yh, ph["gaze_only"], ph["quality_only"])})
res = pd.DataFrame(rows)
print(U.table_md(res))
res.to_csv(OUT / "gaze_quality_auroc.csv", index=False)
tests = pd.DataFrame(tests)
tests["p_holm"] = U.holm(tests.p.values)
print(U.table_md(tests))
tests.to_csv(OUT / "gaze_quality_delong.csv", index=False)

# %% [markdown]
# ## 4. Permutation test (gaze only, and gaze with quality removed)

# %%
rng = np.random.default_rng(0)
perm = {}
for name, resid in (("gaze_only", False), ("gaze_quality_residualised", True)):
    obs = U._safe_auc(y, probs[name])
    null = np.array([U._safe_auc(lab, cv_prob(data, GAZE, resid=resid, repeats=1, labels=lab))
                     for lab in (rng.permutation(y) for _ in range(N_PERM))])
    perm[name] = {"auroc": obs, "null_mean": float(null.mean()), "null_95": float(np.quantile(null, 0.95)),
                  "p_perm": float((1 + (null >= obs).sum()) / (1 + N_PERM)), "n_perm": N_PERM}
    print(name, perm[name])
U.save_json(perm, OUT / "gaze_permutation_test.json")

# %% [markdown]
# ## 5. Gaze features vs tracking ratio

# %%
cor = pd.DataFrame([{"feature": c, "spearman_with_quality": data[[c, Q]].corr("spearman").iloc[0, 1],
                     "spearman_with_label": data[[c, "label"]].corr("spearman").iloc[0, 1]} for c in GAZE])
cor = cor.reindex(cor.spearman_with_quality.abs().sort_values(ascending=False).index)
print(U.table_md(cor.head(15)))
cor.to_csv(OUT / "gaze_feature_quality_correlation.csv", index=False)
print("label vs tracking ratio (Spearman):", round(data[[Q, "label"]].corr("spearman").iloc[0, 1], 3))
