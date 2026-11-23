# %% [meta]
# {"slug": "asd-xmf-g2-gaze-primary", "title": "ASD-XMF G2 Gaze Primary Analysis", "gpu": false, "internet": false,
#  "datasets": [], "kernels": ["asd-xmf-r1-photo-data-gaze"], "phase": "R"}

# %% [markdown]
# # G2 - Pre-specified primary gaze analysis (response to review)
# NB13 and G1 used different fold generators and C values, so the same logistic-regression model was reported
# as 0.943 and 0.949, and the 85 %-sensitivity threshold was chosen on the same out-of-fold predictions it was
# scored on.  G2 fixes the protocol once:
#
# * **primary model** (fixed before running): logistic regression, C = 0.1, median imputation, standardisation,
#   on the 24 non-constant gaze features (the two per-stimulus AOI proportions are constant and are dropped)
# * **identical folds** for every model: repeated stratified 5-fold CV over children, 10 repeats (seeds 0-9)
# * **nested threshold**: inside each training fold an inner 5-fold CV gives out-of-fold training probabilities;
#   the threshold reaching 85 % sensitivity there is applied unchanged to the test fold
# * **quality-adjusted models**: gaze features residualised on the tracking ratio inside each training fold, for
#   the logistic regression, the random forest and LightGBM
# * **tests**: DeLong (Holm over the family), 1000 label permutations of the primary and the residualised model
#   (whole pipeline, 10 repeats each), percentile bootstrap CIs over children
# * subgroup counts reported with every subgroup result

# %%
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "asd_lib"))
import asd_utils as U   # delong_test, bootstrap_ci, holm, threshold_for_sensitivity

SRC = ROOT / "outputs" / "asd-xmf-r1-photo-data-gaze" / "nb13" / "gaze_child_features.csv"
OUT = ROOT / "outputs" / "asd-xmf-g2-gaze-primary"
OUT.mkdir(parents=True, exist_ok=True)
REPEATS, N_PERM, TARGET_SENS = 10, int(os.environ.get("G2_PERM", 1000)), 0.85

d = pd.read_csv(SRC)
y = d.label.values.astype(int)
Q = "all_track_ratio"
meta = {"pid", "sex", "age", "cls", "cars", "label", Q}
GAZE_ALL = [c for c in d.columns if c not in meta]
CONST = [c for c in GAZE_ALL if d[c].nunique(dropna=True) <= 1]
GAZE = [c for c in GAZE_ALL if c not in CONST]
d["sex_m"] = (d.sex == "M").astype(int)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def lr(C):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=5000))


MODELS = {   # name: (feature columns, model factory, residualise gaze on quality?)
    "primary_gaze_lr": (GAZE, lambda: lr(0.1), False),
    "gaze_rf": (GAZE, lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(500, min_samples_leaf=2, random_state=0)), False),
    "quality_only_lr": ([Q], lambda: lr(1.0), False),
    "sex_age_lr": (["sex_m", "age"], lambda: lr(1.0), False),
    "gaze_plus_quality_lr": (GAZE + [Q], lambda: lr(0.1), False),
    "resid_gaze_lr": (GAZE, lambda: lr(0.1), True),
    "resid_gaze_rf": (GAZE, lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(500, min_samples_leaf=2, random_state=0)), True),
}
if lgb is not None:
    mk_lgb = lambda: make_pipeline(SimpleImputer(strategy="median"), lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                                                                         min_child_samples=5, subsample=0.8, subsample_freq=1,
                                                                                         colsample_bytree=0.8, random_state=0, verbose=-1))
    MODELS["gaze_lgbm"] = (GAZE, mk_lgb, False)
    MODELS["resid_gaze_lgbm"] = (GAZE, mk_lgb, True)


def residualise(Xtr, qtr, Xte, qte):
    """Remove the linear effect of tracking quality from every feature, fitted on the training rows only."""
    med = np.nanmedian(Xtr, 0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    R_tr, R_te = Xtr.copy(), Xte.copy()
    for j in range(Xtr.shape[1]):
        m = LinearRegression().fit(qtr[:, None], Xtr[:, j])
        R_tr[:, j] -= m.predict(qtr[:, None]); R_te[:, j] -= m.predict(qte[:, None])
    return R_tr, R_te


def fit_predict(name, tr, te, labels):
    cols, factory, resid = MODELS[name]
    X = d[cols].values.astype(float)
    Xtr, Xte = X[tr], X[te]
    if resid:
        q = d[Q].values.astype(float)
        Xtr, Xte = residualise(Xtr, q[tr], Xte, q[te])
    m = factory().fit(Xtr, labels[tr])
    return m.predict_proba(Xte)[:, 1], (m, Xtr)


def nested_threshold(name, tr, labels, seed):
    """85 %-sensitivity threshold from inner 5-fold out-of-fold probabilities of the training children only."""
    p_in = np.full(len(tr), np.nan)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=1000 + seed).split(tr, labels[tr]):
        p_in[b], _ = fit_predict(name, tr[a], tr[b], labels)
    return U.threshold_for_sensitivity(labels[tr], p_in, TARGET_SENS)


def run(name, labels, repeats=REPEATS, with_threshold=True):
    """Out-of-fold probability per child averaged over repeats, plus per-repeat hard predictions."""
    P = np.zeros((repeats, len(labels))); H = np.zeros((repeats, len(labels)), dtype=int)
    for r in range(repeats):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(np.zeros(len(labels)), labels):
            P[r, te], _ = fit_predict(name, tr, te, labels)
            if with_threshold:
                H[r, te] = (P[r, te] >= nested_threshold(name, tr, labels, r)).astype(int)
    return P.mean(0), H


# %% [markdown]
# ## 1. All models on identical folds, nested thresholds

# %%
t0 = time.time()
res, probs = [], {}
for name in MODELS:
    p, H = run(name, y)
    probs[name] = p
    ci = U.bootstrap_ci(y, p, n_boot=2000, seed=0)
    sens = np.mean([(H[r][y == 1] == 1).mean() for r in range(REPEATS)])
    spec = np.mean([(H[r][y == 0] == 0).mean() for r in range(REPEATS)])
    res.append({"model": name, "n": len(y), "auroc": roc_auc_score(y, p), "ci_lo": ci["lo"], "ci_hi": ci["hi"],
                "sens_nested": sens, "spec_nested": spec})
    print(f"{name:22s} AUROC {roc_auc_score(y, p):.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]  sens {sens:.3f} spec {spec:.3f}")
res = pd.DataFrame(res)
res.to_csv(OUT / "g2_models.csv", index=False)
pd.DataFrame({"pid": d.pid, "label": y, **probs}).to_csv(OUT / "g2_oof_probs.csv", index=False)

# %% [markdown]
# ## 2. DeLong tests (one Holm family)

# %%
pairs = [("primary_gaze_lr", "quality_only_lr"), ("primary_gaze_lr", "sex_age_lr"), ("gaze_plus_quality_lr", "quality_only_lr"),
         ("resid_gaze_lr", "quality_only_lr"), ("resid_gaze_lr", "primary_gaze_lr"), ("resid_gaze_rf", "quality_only_lr")]
if lgb is not None:
    pairs.append(("resid_gaze_lgbm", "quality_only_lr"))
dl = pd.DataFrame([{"model": a, "vs": b, **U.delong_test(y, probs[a], probs[b])} for a, b in pairs])
dl["p_holm"] = U.holm(dl.p.values)
dl.to_csv(OUT / "g2_delong.csv", index=False)
print(dl.round(4).to_string(index=False))

# %% [markdown]
# ## 3. Upper-quality half (counts reported)

# %%
hq = d[Q] >= d[Q].median()
sub = d[hq].reset_index(drop=True); ys = sub.label.values.astype(int)
hq_rows = {"n": int(len(sub)), "n_asd": int(ys.sum()), "n_td": int((1 - ys).sum())}
for name in ("primary_gaze_lr", "quality_only_lr"):
    cols, factory, _ = MODELS[name]
    P = np.zeros((REPEATS, len(ys)))
    k = int(min(5, ys.sum(), (1 - ys).sum()))
    for r in range(REPEATS):
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=r).split(np.zeros(len(ys)), ys):
            P[r, te] = factory().fit(sub[cols].values[tr], ys[tr]).predict_proba(sub[cols].values[te])[:, 1]
    p = P.mean(0); ci = U.bootstrap_ci(ys, p, n_boot=2000, seed=0)
    hq_rows[name] = {"auroc": roc_auc_score(ys, p), "lo": ci["lo"], "hi": ci["hi"]}
    hq_rows[name + "_p"] = p.tolist()
hq_rows["delong"] = U.delong_test(ys, np.array(hq_rows["primary_gaze_lr_p"]), np.array(hq_rows["quality_only_lr_p"]))
for k_ in ("primary_gaze_lr_p", "quality_only_lr_p"):
    hq_rows.pop(k_)
json.dump(hq_rows, open(OUT / "g2_high_quality_half.json", "w"), indent=1, default=float)
print("upper-quality half:", json.dumps(hq_rows, default=float))

# %% [markdown]
# ## 4. Permutation test of the whole pipeline (primary and quality-residualised model)

# %%
perm = {}
rng = np.random.default_rng(0)
for name in ("primary_gaze_lr", "resid_gaze_lr"):
    obs = roc_auc_score(y, probs[name]); null = []
    for i in range(N_PERM):
        yp = rng.permutation(y)
        p, _ = run(name, yp, with_threshold=False)
        null.append(roc_auc_score(yp, p))
    null = np.array(null)
    perm[name] = {"auroc": obs, "n_perm": N_PERM, "null_mean": float(null.mean()), "null_95": float(np.quantile(null, .95)),
                  "p_perm": float((1 + (null >= obs).sum()) / (1 + N_PERM))}
    print(name, perm[name])
json.dump(perm, open(OUT / "g2_permutation.json", "w"), indent=1)

# %% [markdown]
# ## 5. Cohort description (TRIPOD+AI)

# %%
desc = {"n_children": int(len(d)), "n_asd": int(y.sum()), "n_td": int((1 - y).sum()),
        "age_min": float(d.age.min()), "age_max": float(d.age.max()), "age_median": float(d.age.median()),
        "age_median_asd": float(d.age[y == 1].median()), "age_median_td": float(d.age[y == 0].median()),
        "boys_asd": int(((d.sex == "M") & (y == 1)).sum()), "girls_asd": int(((d.sex == "F") & (y == 1)).sum()),
        "boys_td": int(((d.sex == "M") & (y == 0)).sum()), "girls_td": int(((d.sex == "F") & (y == 0)).sum()),
        "cars_asd_median": float(d.cars[y == 1].median()), "cars_td_median": float(d.cars[y == 0].median()),
        "n_gaze_features_used": len(GAZE), "constant_features_dropped": CONST,
        "missing_values_per_feature": {c: int(d[c].isna().sum()) for c in GAZE if d[c].isna().any()},
        "repeats": REPEATS, "target_sensitivity": TARGET_SENS, "minutes": round((time.time() - t0) / 60, 1)}
json.dump(desc, open(OUT / "g2_cohort.json", "w"), indent=1)
print(json.dumps(desc, indent=1))
