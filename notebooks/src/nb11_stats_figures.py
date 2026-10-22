# %% [meta]
# {"slug": "asd-xmf-nb11-stats-figures", "title": "ASD-XMF NB11 Stats Figures",
#  "gpu": false, "internet": true, "datasets": [],
#  "kernels": ["asd-xmf-nb00-data-audit-manifest", "asd-xmf-nb04-frame-baselines-a-cnn", "asd-xmf-nb04-frame-baselines-b-cnn",
#              "asd-xmf-nb04-frame-baselines-c-vit", "asd-xmf-nb04-frame-baselines-d-yolo", "asd-xmf-nb06-video-baselines",
#              "asd-xmf-nb07-audio-baselines", "asd-xmf-nb08-fusion", "asd-xmf-nb09-ablations-shortcuts", "asd-xmf-nb10-xai"],
#  "phase": "B"}

# %% [markdown]
# # NB11 - Statistics, tables and paper figures (CPU only)
# Reads **only** `preds_*.csv` files (plan 8), so no GPU is needed.  Protocol (plan 7):
# * every number per child; threshold fixed on out-of-fold validation predictions at 85 %
#   sensitivity, then applied unchanged to the locked test set
# * 95 % bootstrap CIs (2,000 resamples by child); DeLong (AUROC) and McNemar (paired errors)
#   vs the proposed model, Holm-corrected
# * calibration (ECE, Brier, reliability), decision curves vs **M-CHAT-R/F alone** and
#   **app + M-CHAT**, triage (refer rate + accuracy on the rest), fairness by sex / age band /
#   device / site
# * TRIPOD+AI checklist skeleton written next to the tables

# %%
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
OUT = U.WORK / "nb11"; OUT.mkdir(parents=True, exist_ok=True)
man = pd.read_csv(U.find_path("manifest.csv"))
SYN = bool(man.is_synthetic.iloc[0])
N_BOOT = 300 if PREFLIGHT else 2000
TARGET_SENS = 0.85
PROPOSED = "fusion_xmf_noquest"   # headline model: no questionnaire input (avoids overlap with the reference)
preds = U.load_all_preds([U.INPUT_ROOT])
preds = preds[preds.child_id.isin(man.child_id)]
print("prediction files:", preds.src_file.nunique(), "| models:", preds.model.nunique())

# %% [markdown]
# ## 1. M-CHAT-R/F alone and app + M-CHAT (the clinical bar, plan 5.3)

# %%
from sklearn.linear_model import LogisticRegression
if "mchat_total" in man and man.mchat_total.notna().any():
    rows = []
    for s in sorted(preds.seed.dropna().unique().astype(int)):
        for f in sorted(man[f"fold_s{s}"][man[f"fold_s{s}"] >= 0].unique()):
            tr, va, te = U.fold_ids(man, s, f)
            for part, ids in (("val", va), ("test", te)):
                d = man.set_index("child_id").loc[ids]
                rows.append(pd.DataFrame({"child_id": ids, "y_true": d.label.values, "prob": d.mchat_total.values / 20.0,
                                          "part": part, "model": "MCHAT_alone", "seed": s, "fold": f}))
    mchat = pd.concat(rows)
    combo = []
    app = preds[preds.model == PROPOSED]
    for (s, f), g in app.groupby(["seed", "fold"]):
        tr, va, te = U.fold_ids(man, int(s), int(f))
        oof_tr = app[(app.seed == s) & (app.part == "val") & app.child_id.isin(tr)]
        oof_tr = oof_tr.groupby("child_id").prob.mean()
        if len(oof_tr) < 10:
            continue
        Xtr = np.c_[U.logit(oof_tr.values), man.set_index("child_id").loc[oof_tr.index, "mchat_total"].values]
        lr = LogisticRegression().fit(Xtr, man.set_index("child_id").loc[oof_tr.index, "label"])
        for part in ("val", "test"):
            q = g[g.part == part]
            Xq = np.c_[U.logit(q.prob.values), man.set_index("child_id").loc[q.child_id, "mchat_total"].values]
            combo.append(q.assign(prob=lr.predict_proba(Xq)[:, 1], model="APP_plus_MCHAT"))
    preds = pd.concat([preds, mchat] + combo, ignore_index=True)

# %% [markdown]
# ## 2. Main table: OOF and locked-test metrics with 95 % CIs

# %%
summ, store = [], {}
for m, g in preds.groupby("model"):
    s = U.summarise_model(g, TARGET_SENS, N_BOOT)
    store[m] = s
    for part in ("oof", "test"):
        if f"{part}_ci" not in s:
            continue
        ci = s[f"{part}_ci"].set_index("metric")
        summ.append({"model": m, "part": part, "n": s[part]["n"], "thr": round(s["thr"], 3),
                     **{k: f"{ci.loc[k,'est']:.3f} [{ci.loc[k,'lo']:.3f}, {ci.loc[k,'hi']:.3f}]" for k in ci.index}})
summ = pd.DataFrame(summ)
summ.to_csv(OUT / "table_main_metrics_ci.csv", index=False)
print(U.table_md(summ[summ.part == "test"][["model", "n", "auroc", "auprc", "sens", "spec", "ppv", "npv", "bacc", "ece"]]))

# %% [markdown]
# ## 3. Paired tests vs the proposed model (locked test set), Holm-corrected

# %%
tests = []
if PROPOSED in store and "test_df" in store[PROPOSED]:
    ref = store[PROPOSED]["test_df"].set_index("child_id")
    thr_ref = store[PROPOSED]["thr"]
    for m, s in store.items():
        if m == PROPOSED or "test_df" not in s:
            continue
        j = ref.join(s["test_df"].set_index("child_id"), rsuffix="_o", how="inner")
        if len(j) < 10:
            continue
        dl = U.delong_test(j.y_true, j.prob, j.prob_o)
        mc = U.mcnemar_test(j.y_true, j.prob >= thr_ref, j.prob_o >= s["thr"])
        tests.append({"comparator": m, "n": len(j), "auc_proposed": dl["auc1"], "auc_comparator": dl["auc2"],
                      "delong_p": dl["p"], "mcnemar_p": mc["p"]})
tests = pd.DataFrame(tests)
if len(tests):
    tests["delong_p_holm"] = U.holm(tests.delong_p.values)
    tests["mcnemar_p_holm"] = U.holm(tests.mcnemar_p.values)
    print(U.table_md(tests.sort_values("auc_comparator", ascending=False)))
tests.to_csv(OUT / "table_paired_tests.csv", index=False)

# %% [markdown]
# ## 4. Calibration, decision curves, ROC (figures at 300 dpi)

# %%
show = [m for m in (PROPOSED, "fusion_xmf", "fusion_late_avg_noquest", "fusion_early_concat_noquest", "frame_efficientnet_b0_expr",
                    "video_temporal_transformer", "MCHAT_alone", "APP_plus_MCHAT") if m in store and "test_df" in store[m]]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for m in show:
    t = store[m]["test_df"]
    fpr, tpr, _ = roc_curve(t.y_true, t.prob)
    ax[0].plot(fpr, tpr, label=f"{m} ({U._safe_auc(t.y_true, t.prob):.2f})")
    rc = U.reliability_curve(t.y_true, t.prob, 8); ax[1].plot(rc.p_mean, rc.y_mean, "o-", ms=3, label=m)
    dc = U.decision_curve(t.y_true, t.prob); ax[2].plot(dc.thr, dc.nb_model, label=m)
ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(title="ROC - locked test set", xlabel="1 - specificity", ylabel="sensitivity")
ax[1].plot([0, 1], [0, 1], "k:"); ax[1].set(title="Calibration", xlabel="predicted risk", ylabel="observed ASD rate")
if show:
    ax[2].plot(dc.thr, dc.nb_all, "k--", label="refer all"); ax[2].axhline(0, color="k", lw=0.6)
ax[2].set(title="Decision curve", xlabel="threshold probability", ylabel="net benefit")
for a in ax: a.legend(fontsize=6)
if SYN: fig.suptitle("SYNTHETIC DEV COHORT - NOT RESULTS", color="red")
plt.tight_layout(); plt.savefig(OUT / "fig_roc_calibration_dca.png", dpi=300); plt.show()

# %% [markdown]
# ## 5. Triage: refer the most uncertain children (evidential uncertainty)

# %%
tri = []
if PROPOSED in store and "test_df" in store[PROPOSED]:
    t = store[PROPOSED]["test_df"].dropna(subset=["uncertainty"])
    for q in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        tri.append({"refer_quantile": q, **U.triage_metrics(t.y_true, t.prob, t.uncertainty, store[PROPOSED]["thr"], q)})
tri = pd.DataFrame(tri)
print(U.table_md(tri[[c for c in tri.columns if c in ("refer_quantile", "refer_rate", "kept_n", "kept_auroc", "kept_sens", "kept_spec", "kept_bacc")]]) if len(tri) else "no uncertainty")
tri.to_csv(OUT / "table_triage.csv", index=False)

# %% [markdown]
# ## 6. Fairness (sex, age band, device, site) for the proposed model

# %%
if PROPOSED in store and "test_df" in store[PROPOSED]:
    t = store[PROPOSED]["test_df"].merge(man[["child_id", "sex", "age_band", "device", "site"]], on="child_id")
    fair = U.fairness_table(t, ["sex", "age_band", "device", "site"], store[PROPOSED]["thr"])
    print(U.table_md(fair)); fair.to_csv(OUT / "table_fairness.csv", index=False)

# %% [markdown]
# ## 7. Ablation / shortcut / robustness table and TRIPOD+AI skeleton

# %%
abl = summ[(summ.part == "oof") & summ.model.str.startswith(("ablation_", "SHORTCUT_", "fusion_xmf"))][["model", "auroc", "bacc", "ece"]]
print(U.table_md(abl)); abl.to_csv(OUT / "table_ablations_shortcuts.csv", index=False)
tripod = """# TRIPOD+AI checklist - pointers (fill page numbers in the manuscript)
1 Title: screening (not diagnosis) model, children 18-72 months, Bangladesh
3 Background / 4 Objectives: screening vs M-CHAT-R/F; decision support only
5 Data sources: prospective protocol recordings, sites, dates  (NB00 audit_report.json)
6 Participants: eligibility, ASD / TD / DD groups, consent
7 Data preparation: NB01 / NB02 pipelines, missing-modality handling
8 Outcome: clinician DSM-5 (+ADOS-2) diagnosis blind to the model
9 Predictors: face video, voice, cry, M-CHAT items (reported with/without)
10 Sample size: statistician calculation (attach)
11 Missing data: learned missing tokens + modality dropout
12 Analysis: child-wise CV, locked test (NB00), threshold at 85 % sensitivity on OOF
13 Class imbalance / 14 Fairness: table_fairness.csv
15 Model output: risk + uncertainty, 3-way decision (table_triage.csv)
16 Training vs evaluation: table_main_metrics_ci.csv (OOF vs locked test)
20 Performance: AUROC/AUPRC/sens/spec/PPV/NPV/calibration with 95 % CIs
22 Model updating / 23 interpretability: NB10 outputs
25 Shortcut / confounding analysis: table_ablations_shortcuts.csv
27 Code & data availability: Kaggle notebooks + asd_lib; data per consent
"""
(OUT / "TRIPOD_AI_checklist_skeleton.md").write_text(tripod)
if SYN:
    print(U.SYNTHETIC_BANNER)
