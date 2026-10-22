# %% [meta]
# {"slug": "asd-xmf-nb13-gaze-eyetracking", "title": "ASD-XMF NB13 Gaze Eyetracking",
#  "gpu": false, "internet": true,
#  "datasets": ["imtkaggleteam/eye-tracking-autism"], "kernels": [], "phase": "A"}

# %% [markdown]
# # NB13 - Gaze module + external sanity check (Cilia et al. 2022 eye-tracking data)
# Plan section 4.2: *"Gaze-feature module, external sanity check."*  59 children
# (29 ASD, 30 TD, CARS-scored) watched video stimuli on an SMI eye tracker.
# This is the one public ASD-vs-TD set with consent and clinical labels that we can
# analyse **today**, so it also rehearses the full evaluation protocol of section 7:
#
# * child-level features only, repeated stratified 5-fold CV by child (3 seeds)
# * 95 % bootstrap CIs, DeLong vs baselines, calibration, decision curves
# * **shortcut baselines**: sex + age only, and data-quality only (tracking ratio).
#   In this cohort ASD is 83 % male vs 43 % in TD, so sex alone predicts the label -
#   the gaze model must beat that, and we also report a males-only analysis.
# * SHAP on gaze features (behaviour-level XAI)
#
# Cite: Cilia F. et al., *Eye-tracking dataset to support the research on autism spectrum
# disorder*, 2022 (figshare / Kaggle mirror `imtkaggleteam/eye-tracking-autism`).

# %%
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = U.WORK / "nb13"; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (0, 1, 2)
N_BOOT = 300 if PREFLIGHT else 2000

meta_p = U.find_path("Metadata_Participants.csv")
csv_dir = U.find_path("Eye-tracking Output", want="dir")
meta = pd.read_csv(meta_p)
meta.columns = [c.strip() for c in meta.columns]
meta = meta.rename(columns={"ParticipantID": "pid", "Gender": "sex", "Age": "age", "Class": "cls", "CARS Score": "cars"})
meta["label"] = (meta.cls.str.upper() == "ASD").astype(int)
print(meta.groupby("cls").agg(n=("pid", "size"), male=("sex", lambda s: (s == "M").mean()), age=("age", "mean")))

# %% [markdown]
# ## 1. Event-level parsing -> child-level gaze features

# %%
USE = ["RecordingTime [ms]", "Stimulus", "Trial", "Participant", "Tracking Ratio [%]", "Category Right", "Index Right",
       "Pupil Diameter Right [mm]", "Pupil Diameter Left [mm]", "Point of Regard Right X [px]",
       "Point of Regard Right Y [px]", "AOI Name Right"]
files = sorted(csv_dir.glob("*.csv"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
if PREFLIGHT:
    files = files[:4]
parts = []
for f in files:
    d = pd.read_csv(f, usecols=lambda c: c in USE, low_memory=False)
    d["file"] = f.stem
    parts.append(d)
raw = pd.concat(parts, ignore_index=True)
raw = raw[pd.to_numeric(raw.Participant, errors="coerce").notna()].copy()
raw["pid"] = raw.Participant.astype(float).astype(int)
for c in ["RecordingTime [ms]", "Tracking Ratio [%]", "Pupil Diameter Right [mm]", "Pupil Diameter Left [mm]",
          "Point of Regard Right X [px]", "Point of Regard Right Y [px]", "Index Right"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw["stim"] = raw.Stimulus.astype(str).str.lower().str.replace(r"\.(avi|mp4|jpg|png)$", "", regex=True).str.strip()
print(raw.shape, raw.pid.nunique(), "participants;", raw.stim.nunique(), "stimuli")

def child_features(g: pd.DataFrame) -> dict:
    t = g["RecordingTime [ms]"]
    dur_min = max(1e-3, (t.max() - t.min()) / 60000)
    cat = g["Category Right"].astype(str)
    ev = g.assign(cat=cat).groupby(["Trial", "Index Right", "cat"])["RecordingTime [ms]"].agg(["min", "max", "size"]).reset_index()
    fix = ev[ev.cat == "Fixation"]
    sac = ev[ev.cat == "Saccade"]
    fx_d = (fix["max"] - fix["min"]).clip(lower=0)
    por = g.loc[cat == "Fixation", ["Point of Regard Right X [px]", "Point of Regard Right Y [px]"]].dropna()
    pup = g[["Pupil Diameter Right [mm]", "Pupil Diameter Left [mm]"]].replace(0, np.nan).mean(1)
    aoi = g["AOI Name Right"].astype(str)
    on_aoi = (aoi != "-") & (aoi != "nan") & (aoi.str.lower() != "white space")
    return {
        "fix_prop": (cat == "Fixation").mean(), "sac_prop": (cat == "Saccade").mean(), "blink_prop": (cat == "Blink").mean(),
        "fix_rate": len(fix) / dur_min, "sac_rate": len(sac) / dur_min,
        "blink_rate": ev[ev.cat == "Blink"].shape[0] / dur_min,
        "fix_dur_mean": fx_d.mean(), "fix_dur_median": fx_d.median(), "fix_dur_sd": fx_d.std(),
        "gaze_disp_x": por.iloc[:, 0].std() if len(por) else np.nan, "gaze_disp_y": por.iloc[:, 1].std() if len(por) else np.nan,
        "pupil_mean": pup.mean(), "pupil_sd": pup.std(), "aoi_on_prop": on_aoi.mean(),
        "track_ratio": g["Tracking Ratio [%]"].mean(),
    }

feats_all = raw.groupby("pid").apply(child_features).apply(pd.Series)
feats_all.columns = ["all_" + c for c in feats_all.columns]
# per-stimulus features for stimuli seen by >= 80 % of children
cover = raw.groupby("stim").pid.nunique()
common = cover[cover >= 0.8 * raw.pid.nunique()].index.tolist()[:12]
print("common stimuli:", common)
per = []
for s in common:
    f = raw[raw.stim == s].groupby("pid").apply(child_features).apply(pd.Series)
    f = f[["fix_prop", "fix_dur_mean", "gaze_disp_x", "gaze_disp_y", "aoi_on_prop", "blink_rate"]]
    f.columns = [f"{s[:24]}__{c}" for c in f.columns]
    per.append(f)
F = pd.concat([feats_all] + per, axis=1)
data = meta.set_index("pid").join(F, how="inner").reset_index()
QUALITY = ["all_track_ratio"]
GAZE = [c for c in F.columns if c not in QUALITY]
print("children with gaze:", len(data), "| gaze features:", len(GAZE))
data.to_csv(OUT / "gaze_child_features.csv", index=False)

# %% [markdown]
# ## 2. Repeated child-wise CV for gaze models and shortcut baselines

# %%
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
data["child_id"] = data.pid.astype(str)
data["sex_m"] = (data.sex == "M").astype(int)
splits = U.make_child_splits(data[["child_id", "label", "sex"]], n_splits=5, seeds=SEEDS, test_frac=0.0)
data = data.merge(splits[["child_id", "split"] + [f"fold_s{s}" for s in SEEDS]], on="child_id")

def lr(C=0.3): return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=5000))
MODELS = {
    "gaze_logreg": (GAZE, lambda: lr(0.1)),
    "gaze_svm": (GAZE, lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(C=1.0, probability=True))),
    "gaze_rf": (GAZE, lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(500, min_samples_leaf=2, random_state=0))),
    "gaze_et": (GAZE, lambda: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=2, random_state=0))),
    "gaze_plus_demo_logreg": (GAZE + ["sex_m", "age"], lambda: lr(0.1)),
    "SHORTCUT_sex_age": (["sex_m", "age"], lambda: lr(1.0)),
    "SHORTCUT_quality": (QUALITY, lambda: lr(1.0)),
}
try:
    import lightgbm as lgb
    MODELS["gaze_lgbm"] = (GAZE, lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                                            min_child_samples=5, subsample=0.8, colsample_bytree=0.7, verbose=-1))
except Exception:
    pass

oof_rows = []
for name, (cols, mk) in MODELS.items():
    for s in SEEDS:
        for f in sorted(data[f"fold_s{s}"].unique()):
            tr, va = data[data[f"fold_s{s}"] != f], data[data[f"fold_s{s}"] == f]
            U.assert_no_leak(tr.child_id, va.child_id)
            m = mk().fit(tr[cols], tr.label)
            p = m.predict_proba(va[cols])[:, 1]
            d = pd.DataFrame({"child_id": va.child_id, "y_true": va.label, "prob": p, "part": "val"})
            U.save_preds(d, name, f, OUT / "preds", seed=s)
            oof_rows.append(d.assign(model=name, seed=s, fold=f))
oof = pd.concat(oof_rows)

# %% [markdown]
# ## 3. Results with 95 % CIs (pooled over 3 seeds, threshold at 85 % sensitivity)

# %%
summ, curves = [], {}
for name in MODELS:
    s = U.summarise_model(oof[oof.model == name], target_sens=0.85, n_boot=N_BOOT)
    ci = s["oof_ci"].set_index("metric")
    row = {"model": name, "n_children": s["oof"]["n"]}
    for k in ("auroc", "auprc", "sens", "spec", "bacc", "brier", "ece"):
        row[k] = f"{ci.loc[k,'est']:.3f} [{ci.loc[k,'lo']:.3f}-{ci.loc[k,'hi']:.3f}]"
    summ.append(row)
    curves[name] = oof[oof.model == name].groupby("child_id").agg(y=("y_true", "first"), p=("prob", "mean"))
summ = pd.DataFrame(summ)
print(U.table_md(summ))
summ.to_csv(OUT / "gaze_results_ci.csv", index=False)

# DeLong: every model vs the sex+age shortcut, Holm-corrected
ref = curves["SHORTCUT_sex_age"]
tests = []
for name, c in curves.items():
    if name == "SHORTCUT_sex_age":
        continue
    j = c.join(ref, rsuffix="_ref")
    tests.append({"model": name, **U.delong_test(j.y, j.p, j.p_ref)})
tests = pd.DataFrame(tests)
tests["p_holm"] = U.holm(tests.p.values)
print(U.table_md(tests))
tests.to_csv(OUT / "gaze_delong_vs_sex_age.csv", index=False)

# %% [markdown]
# ## 4. Confound control: males only, and fairness by sex / age

# %%
best = max([m for m in curves if m.startswith("gaze_")], key=lambda m: U._safe_auc(curves[m].y, curves[m].p))
cb = curves[best].join(data.set_index("child_id")[["sex", "age"]])
male = cb[cb.sex == "M"]
print(f"best gaze model: {best}")
print("males only AUROC:", U.fmt_ci(U.bootstrap_ci(male.y.values, male.p.values, n_boot=N_BOOT)), f"(n={len(male)})")
cb["age_band"] = pd.cut(cb.age, [0, 6, 9, 20], labels=["<6y", "6-9y", "9y+"])
fair = U.fairness_table(cb.rename(columns={"y": "y_true", "p": "prob"}), ["sex", "age_band"], thr=0.5)
print(U.table_md(fair))
fair.to_csv(OUT / "gaze_fairness.csv", index=False)

# %% [markdown]
# ## 5. Calibration, decision curve, ROC figure

# %%
from sklearn.metrics import roc_curve
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for name in [best, "gaze_plus_demo_logreg", "SHORTCUT_sex_age", "SHORTCUT_quality"]:
    c = curves[name]
    fpr, tpr, _ = roc_curve(c.y, c.p)
    ax[0].plot(fpr, tpr, label=f"{name} ({U._safe_auc(c.y, c.p):.2f})")
    rc = U.reliability_curve(c.y, c.p, 5)
    ax[1].plot(rc.p_mean, rc.y_mean, "o-", label=name)
    dc = U.decision_curve(c.y, c.p)
    ax[2].plot(dc.thr, dc.nb_model, label=name)
ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(title="ROC (OOF, child-level)", xlabel="1 - specificity", ylabel="sensitivity")
ax[1].plot([0, 1], [0, 1], "k:"); ax[1].set(title="Reliability", xlabel="predicted", ylabel="observed")
ax[2].plot(dc.thr, dc.nb_all, "k--", label="refer all"); ax[2].axhline(0, color="k", lw=0.5)
ax[2].set(title="Decision curve", xlabel="threshold probability", ylabel="net benefit", ylim=(-0.1, 0.6))
for a in ax: a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "gaze_roc_calib_dca.png", dpi=200); plt.show()

# %% [markdown]
# ## 6. SHAP on gaze features (behaviour-level explanation)

# %%
try:
    import shap
    cols = GAZE
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(data[cols], data.label)
    Xt = model[:-1].transform(data[cols])
    ex = shap.LinearExplainer(model[-1], Xt)
    sv = ex.shap_values(Xt)
    imp = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(sv).mean(0), "coef": model[-1].coef_[0]}).sort_values("mean_abs_shap", ascending=False)
    print(U.table_md(imp.head(15)))
    imp.to_csv(OUT / "gaze_shap_importance.csv", index=False)
    shap.summary_plot(sv, pd.DataFrame(Xt, columns=cols), show=False, max_display=15)
    plt.tight_layout(); plt.savefig(OUT / "gaze_shap_summary.png", dpi=200); plt.show()
except Exception as e:
    print("SHAP skipped:", e)

# %%
U.save_json({"best_model": best, "n_children": int(len(data)), "n_gaze_features": len(GAZE),
             "note": "Repeated 5-fold CV by child over all 59 children (no separate test set: n too small)."},
            OUT / "nb13_summary.json")
print(sorted(p.name for p in OUT.iterdir()))
