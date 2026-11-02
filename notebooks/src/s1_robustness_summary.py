# %% [meta]
# {"slug": "asd-xmf-s1-robustness-summary", "title": "ASD-XMF S1 Robustness Summary",
#  "gpu": false, "internet": true, "datasets": [],
#  "kernels": ["asd-xmf-r1-photo-data-gaze", "asd-xmf-r2-photo-seeds", "asd-xmf-r3-audio-text-video-seeds"], "phase": "R"}

# %% [markdown]
# # S1 - Robustness summary (R1-R3) and comparison with published results
# * every sector over 5 seeds: mean +- SD, how often the proposed model wins, paired t-test over seeds (Holm)
# * photo: near-duplicate audit and accuracy on the clean test set
# * gaze: 10-repeat child-wise CV and the quality checks (G1)
# * published results on the same public datasets.  Only numbers checked against the original paper
#   are filled in; protocols differ (official splits, extra data, ensembles), so the table is context,
#   not a head-to-head ranking.

# %%
import re
import numpy as np, pandas as pd
from scipy import stats
OUT = U.WORK / "s1"; OUT.mkdir(parents=True, exist_ok=True)
md = ["# Robustness summary (R1-R4)\n"]

def seed_runs(fname, prefix):
    """All <prefix>_s<k>/<fname> files -> one frame with a seed column."""
    parts = []
    for p in U.find_all(fname):
        m = re.fullmatch(prefix + r"_s(\d+)", p.parent.name)
        if m:
            parts.append(pd.read_csv(p).assign(seed=int(m.group(1))))
    return pd.concat(parts, ignore_index=True).drop_duplicates(["task", "model", "seed"]) if parts else pd.DataFrame()

def sweep_tables(df, metrics, sector):
    """Mean +- SD per task/model and paired tests proposed vs each baseline over seeds."""
    summ = df.groupby(["task", "model"])[metrics].agg(["mean", "std", "count"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ["proposed"] = summ.model.isin(df.loc[df.proposed.astype(bool), "model"].unique())
    summ = summ.sort_values(["task", f"{metrics[0]}_mean"], ascending=[True, False])
    pair = []
    for task, g in df.groupby("task"):
        prop = g.loc[g.proposed.astype(bool), "model"].iloc[0]
        for b in sorted(set(g.model) - {prop}):
            for met in metrics:
                a_ = g[g.model == prop].set_index("seed")[met]; b_ = g[g.model == b].set_index("seed")[met]
                c = a_.index.intersection(b_.index)
                d = (a_[c] - b_[c]).values
                p = stats.ttest_rel(a_[c], b_[c]).pvalue if len(c) > 1 and np.std(d) > 0 else np.nan
                pair.append({"task": task, "proposed": prop, "baseline": b, "metric": met, "n_seeds": len(c),
                             "mean_diff": d.mean() if len(d) else np.nan, "wins": int((d > 0).sum()), "p_paired_t": p})
    pair = pd.DataFrame(pair)
    if len(pair):
        pair["p_holm"] = U.holm(pair.p_paired_t.fillna(1).values)
    summ.to_csv(OUT / f"{sector}_seeds_summary.csv", index=False)
    pair.to_csv(OUT / f"{sector}_seeds_paired_tests.csv", index=False)
    md.extend([f"\n## {sector.capitalize()} over seeds\n", U.table_md(summ, ".4f"),
               f"\n\nPaired tests ({sector}, proposed vs baseline, over seeds):\n", U.table_md(pair, ".4f"), "\n"])
    print(f"== {sector}\n", U.table_md(summ, ".4f"))
    return summ, pair

# %% [markdown]
# ## Audio, text, video (R3)

# %%
audio = seed_runs("audio_comparison.csv", "a1")
text = seed_runs("text_comparison.csv", "t1")
video = seed_runs("video_comparison.csv", "v1")
if len(audio):
    sweep_tables(audio, ["test_macro_f1", "test_bal_acc", "test_acc"], "audio")
if len(text):
    sweep_tables(text, ["test_acc", "test_macro_f1"], "text")
if len(video):
    sweep_tables(video, ["test_acc", "test_macro_f1"], "video")

# %% [markdown]
# ## Photo (R2) and near-duplicate audit (R1)

# %%
for name, title in (("near_dup_summary.csv", "Near-duplicate audit (test vs train+val, pHash Hamming distance)"),
                    ("photo_seeds_summary.csv", "Photo over seeds (accuracy on full and clean test set)"),
                    ("photo_seeds_paired_tests.csv", "Photo paired tests over seeds (XMF-Face v2 vs baseline)")):
    p = U.find_path(name)
    if p:
        t = pd.read_csv(p); t.to_csv(OUT / name, index=False)
        md.extend([f"\n## {title}\n", U.table_md(t, ".4f"), "\n"])
        print(f"== {title}\n", U.table_md(t, ".4f"))
    else:
        print("missing", name)

# %% [markdown]
# ## Gaze (R1): 10-repeat CV and quality checks

# %%
for name, title in (("gaze_results_ci.csv", "Gaze models and shortcut baselines (10 repeats of child-wise 5-fold CV)"),
                    ("gaze_quality_auroc.csv", "Gaze vs recording quality"),
                    ("gaze_quality_delong.csv", "DeLong tests: does gaze add to quality?")):
    p = U.find_path(name)
    if p:
        t = pd.read_csv(p); t.to_csv(OUT / name, index=False)
        md.extend([f"\n## {title}\n", U.table_md(t), "\n"])
        print(f"== {title}\n", U.table_md(t))
p = U.find_path("gaze_permutation_test.json")
if p:
    md.extend(["\n## Gaze permutation test\n", "```\n" + p.read_text() + "\n```\n"])

# %% [markdown]
# ## Published results on the same datasets (context only)

# %%
# verified = checked against the original paper's abstract/text; protocols are not identical to ours
PUBLISHED = [
    {"dataset": "FER2013 (official test)", "task": "7-class expression", "metric": "accuracy",
     "published": 0.7328, "method": "VGGNet, single network, no extra data",
     "source": "Khaireddin & Chen 2021, arXiv:2105.03588", "verified": True},
    {"dataset": "RAF-DB (official test)", "task": "7-class expression", "metric": "accuracy",
     "published": 0.9221, "method": "POSTER++ (ViT + landmark cross-fusion, 224 px, pre-trained on MS-Celeb)",
     "source": "Mao et al. 2023, arXiv:2301.12149", "verified": True},
    {"dataset": "MMASD (activity)", "task": "11-class activity", "metric": "accuracy", "published": np.nan,
     "method": "no baseline in the dataset paper (Li et al. 2023, arXiv:2306.08243)", "source": "-", "verified": True},
    {"dataset": "TASD (text)", "task": "ASD vs non-ASD", "metric": "accuracy", "published": np.nan,
     "method": "no peer-reviewed benchmark found", "source": "-", "verified": False},
    {"dataset": "ReCANVo (vocal affect)", "task": "vocalisation label", "metric": "macro-F1", "published": np.nan,
     "method": "per-participant results in Johnson et al. 2023 (Sci Data 10) - protocol differs, number not checked", "source": "-", "verified": False},
    {"dataset": "Cilia/Carette eye-tracking (59 children)", "task": "ASD vs TD", "metric": "AUROC", "published": np.nan,
     "method": "scanpath images + CNN (Carette et al. 2019; Cilia et al. 2021 JMIR Hum Factors) - image-level, number not checked",
     "source": "-", "verified": False},
]
pub = pd.DataFrame(PUBLISHED)
ours = {}
p = U.find_path("photo_seeds_summary.csv")
if p:
    ps = pd.read_csv(p).set_index("model")
    if "xmf_face" in ps.index:
        ours["FER2013 (official test)"] = f"{ps.loc['xmf_face', 'acc_fer2013_mean']:.4f} +- {ps.loc['xmf_face', 'acc_fer2013_std']:.4f}"
        ours["RAF-DB (official test)"] = f"{ps.loc['xmf_face', 'acc_rafdb_mean']:.4f} +- {ps.loc['xmf_face', 'acc_rafdb_std']:.4f}"
if len(video):
    v = video[video.proposed.astype(bool)].test_acc
    ours["MMASD (activity)"] = f"{v.mean():.4f} +- {v.std():.4f} (child-wise 5-fold)"
if len(text):
    t = text[text.proposed.astype(bool)].test_acc
    ours["TASD (text)"] = f"{t.mean():.4f} +- {t.std():.4f}"
if len(audio):
    a = audio[audio.proposed.astype(bool) & (audio.task == "vocal_affect")].test_macro_f1
    ours["ReCANVo (vocal affect)"] = f"{a.mean():.4f} +- {a.std():.4f} (leave-one-speaker-out)"
gp = U.find_path("gaze_quality_auroc.csv")
if gp:
    g = pd.read_csv(gp)
    ours["Cilia/Carette eye-tracking (59 children)"] = g.loc[g.model == "gaze_only", "auroc"].iloc[0] + " (child-level)"
pub["ours_mean_sd"] = pub.dataset.map(ours).fillna("-")
print(U.table_md(pub, ".4f"))
pub.to_csv(OUT / "published_comparison.csv", index=False)
md.extend(["\n## Published results on the same datasets (context, protocols differ)\n", U.table_md(pub, ".4f"), "\n"])
(OUT / "robustness_summary.md").write_text("\n".join(md), encoding="utf-8")
print("written", OUT / "robustness_summary.md")
