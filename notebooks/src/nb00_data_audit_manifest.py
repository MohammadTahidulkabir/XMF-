# %% [meta]
# {"slug": "asd-xmf-nb00-data-audit-manifest", "title": "ASD-XMF NB00 Data Audit Manifest",
#  "gpu": false, "internet": true,
#  "datasets": ["shuvoalok/raf-db-dataset"], "kernels": [], "phase": "B"}

# %% [markdown]
# # NB00 - Cohort audit, duplicate check and locked child-wise splits
# **Input.** The BanglaASD-XMF cohort as a private Kaggle dataset (attach it under
# *Add Input*) with this layout - nothing else is required:
# ```
# metadata/children.csv  child_id,label(1=ASD),group(ASD/TD/DD),age_months,sex,site,device,room,recorder,
#                        mchat_total,mchat_01..mchat_20
# metadata/clips.csv     child_id,clip_id,stimulus,video_path[,audio_path]   (paths relative to the dataset root)
# metadata/cry.csv       child_id,cry_id,cry_path                            (natural cries only - never provoked)
# ```
# If consent does not allow raw video in the cloud, run NB01/NB02 locally and upload only
# their feature outputs; every later notebook reads features, not video.
#
# **No cohort attached -> DEV mode.** A small *synthetic* cohort is generated from RAF-DB
# faces and synthetic tones, purely so NB01-NB12 can be executed and debugged before
# ethics approval.  Every DEV output is stamped SYNTHETIC; none of it is a result.
#
# **Output.** `manifest.csv` (one row per child + locked `split` + `fold_s0..2`),
# `clips.csv` with `clip_idx`, `cry.csv`, the audit report and the pHash duplicate report.

# %%
import numpy as np, pandas as pd
from pathlib import Path
import cv2
OUT = U.WORK / "nb00"; OUT.mkdir(parents=True, exist_ok=True)
EXTERNAL_SITE = None      # e.g. "site2" -> that whole site becomes the locked test set (preferred)
TEST_FRAC = 0.20          # used when no external site is given
SEEDS = (0, 1, 2)
N_SPLITS = 5

root, is_dev = U.locate_cohort()
if root is None:
    raf = U.find_path("DATASET", want="dir")
    happy = sorted((raf / "train" / "4").glob("*.jpg"))
    neutral = sorted((raf / "train" / "7").glob("*.jpg"))
    spec = U.DevCohortSpec(n_children=16 if PREFLIGHT else 60, clips_per_child=3 if PREFLIGHT else 6)
    root = U.make_dev_cohort(OUT / "dev_cohort", happy, neutral, spec)
    is_dev = True
print("cohort root:", root, "| DEV:", is_dev)
if is_dev:
    print(U.SYNTHETIC_BANNER)

# %% [markdown]
# ## 1. Schema checks

# %%
children = pd.read_csv(root / "metadata" / "children.csv")
clips = pd.read_csv(root / "metadata" / "clips.csv")
cry = pd.read_csv(root / "metadata" / "cry.csv") if (root / "metadata" / "cry.csv").exists() else pd.DataFrame(columns=["child_id", "cry_id", "cry_path"])
problems = []
need = {"child_id", "label", "age_months", "sex"}
miss = need - set(children.columns)
if miss: problems.append(f"children.csv missing columns {miss}")
if children.child_id.duplicated().any(): problems.append("duplicate child_id in children.csv")
if not set(children.label.unique()) <= {0, 1}: problems.append("label must be 0/1 (1 = ASD)")
orph = set(clips.child_id) - set(children.child_id)
if orph: problems.append(f"{len(orph)} clip rows reference unknown children")
clips["exists"] = [(root / p).exists() for p in clips.video_path]
if not clips.exists.all(): problems.append(f"{(~clips.exists).sum()} video files missing")
if len(cry):
    cry["exists"] = [(root / p).exists() for p in cry.cry_path]
for c in ("site", "device", "room", "recorder", "group"):
    if c not in children:
        children[c] = "unknown"
print("schema problems:", problems or "none")
clips["clip_idx"] = clips.groupby("child_id").cumcount()   # shared clip order for NB01 and NB02

# %% [markdown]
# ## 2. Cohort description and confound audit
# Device / room / recorder / site must be **independent of the label** - otherwise a model
# can learn the room instead of the child (plan 5.4).  Age and sex imbalance is reported
# and later handled in fairness and matched analyses.

# %%
from scipy import stats
def cramers_v(tab):
    chi2 = stats.chi2_contingency(tab)[0]
    n = tab.values.sum(); r, k = tab.shape
    return float(np.sqrt(chi2 / (n * max(1, min(r, k) - 1))))
desc = children.groupby("label").agg(n=("child_id", "size"), age_mean=("age_months", "mean"), age_sd=("age_months", "std"),
                                     male=("sex", lambda s: (s == "M").mean()))
print(desc)
audit = []
for col in ("sex", "site", "device", "room", "recorder", "group"):
    tab = pd.crosstab(children[col], children.label)
    if tab.shape[0] > 1:
        p = stats.chi2_contingency(tab)[1]
        audit.append({"variable": col, "test": "chi2", "p": p, "cramers_v": cramers_v(tab),
                      "flag": "CONFOUND RISK" if (col in ("site", "device", "room", "recorder") and p < 0.05) else ""})
t = stats.mannwhitneyu(children.age_months[children.label == 1], children.age_months[children.label == 0])
audit.append({"variable": "age_months", "test": "mann-whitney", "p": t.pvalue, "cramers_v": np.nan, "flag": ""})
audit = pd.DataFrame(audit)
print(U.table_md(audit))
clip_per_child = clips.groupby("child_id").size()
print("clips per child:", clip_per_child.describe().round(1).to_dict())
print("children with cry:", cry.child_id.nunique() if len(cry) else 0)

# %% [markdown]
# ## 3. Perceptual-hash duplicate check (before any split)
# Middle frame of every clip.  Near-identical frames from **different children** mean a
# file was copied or mislabelled and must be fixed at the source before training.

# %%
def middle_frame(path):
    cap = cv2.VideoCapture(str(path)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2); ok, fr = cap.read(); cap.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None
hashes, keys, owners = [], [], {}
for r in clips[clips.exists].itertuples():
    fr = middle_frame(root / r.video_path)
    if fr is not None:
        hashes.append(U.phash(fr)); keys.append(r.clip_id); owners[r.clip_id] = r.child_id
dups = U.near_duplicates(hashes, keys, max_dist=4)
if len(dups):
    dups["child_a"] = dups.a.map(owners); dups["child_b"] = dups.b.map(owners)
    cross = dups[dups.child_a != dups.child_b]
else:
    cross = dups
print(f"near-duplicate clip pairs: {len(dups)} | across different children: {len(cross)}")
cross.to_csv(OUT / "phash_cross_child_duplicates.csv", index=False)
if len(cross) and not is_dev:
    problems.append(f"{len(cross)} cross-child duplicate clips - resolve before training")

# %% [markdown]
# ## 4. Locked splits (by child) - created once, read by every notebook

# %%
man = U.make_child_splits(children, n_splits=N_SPLITS, seeds=SEEDS, test_frac=TEST_FRAC,
                          external_site=EXTERNAL_SITE, site_col="site")
for s in SEEDS:
    for f in range(N_SPLITS):
        tr, va, te = U.fold_ids(man, s, f)
        U.assert_no_leak(tr, va, te)
man["age_band"] = U.age_band(man.age_months).astype(str)
man["is_synthetic"] = is_dev
man["cohort_root"] = str(root)
print(man.groupby(["split", "label"]).size().unstack())
man.to_csv(OUT / "manifest.csv", index=False)
clips.to_csv(OUT / "clips.csv", index=False)
cry.to_csv(OUT / "cry.csv", index=False)
report = {"is_synthetic": bool(is_dev), "n_children": int(len(man)), "n_clips": int(len(clips)),
          "n_cry": int(len(cry)), "problems": problems, "audit": audit.to_dict("records"),
          "test_children": int((man.split == "test").sum()), "external_site": EXTERNAL_SITE}
U.save_json(report, OUT / "audit_report.json")
print(json.dumps({k: v for k, v in report.items() if k != "audit"}, indent=1))
assert not [p for p in problems if "missing columns" in p or "label must" in p], problems
