# %% [meta]
# {"slug": "asd-xmf-nb01-face-pipeline", "title": "ASD-XMF NB01 Face Pipeline",
#  "gpu": false, "internet": true,
#  "datasets": [], "kernels": ["asd-xmf-nb00-data-audit-manifest"], "phase": "B"}

# %% [markdown]
# # NB01 - Face pipeline: decode once -> aligned crops + behaviour features
# Each protocol clip is decoded **once** at 5 fps (max 40 frames = 8 s).  MediaPipe Face
# Landmarker gives 478 landmarks, 52 blendshapes (action-unit proxies), head pose and iris
# position; the face is eye-aligned to a 160 px crop.  Saved per child as uint8 `.npy`
# shards that NB04-NB10 memory-map - no notebook ever decodes video again.
#
# Runs on **CPU** (all cores, one landmarker per process): costs no GPU quota.
# For the real cohort attach the cohort dataset in addition to NB00's output.
#
# Outputs (`/kaggle/working/face`): `crops/ valid/ hand/ boxes/ qual/ bg/ rows/` + `face_index.csv`
# + `face_qc.csv`.  `bg/` holds face-masked background thumbnails for the shortcut test.

# %%
import numpy as np, pandas as pd
from pathlib import Path
import asd_face as AF
U.pip_install("mediapipe")
FPS, MAX_T, CROP, N_BG = 5, 40, 160, 4
OUT = U.WORK / "face"; OUT.mkdir(parents=True, exist_ok=True)
man = pd.read_csv(U.find_path("manifest.csv"))
clips = pd.read_csv(U.find_path("manifest.csv").parent / "clips.csv")
root, is_dev = U.locate_cohort()
print("cohort:", root, "DEV" if is_dev else "REAL", "| children", len(man), "| clips", len(clips))
if PREFLIGHT:
    man = man.head(6)
model_path = AF.get_landmarker_model(U.WORK / "face_landmarker.task")

# %%
jobs = []
for cid in man.child_id:
    rows = clips[clips.child_id == cid].sort_values("clip_idx").to_dict("records")
    if rows:
        jobs.append((cid, rows, str(root), str(OUT), FPS, MAX_T, CROP, N_BG))
import multiprocessing as mp
results, failed = [], []
ctx = mp.get_context("spawn")   # fork + OpenCV/MediaPipe threads can deadlock on Linux
with U.Timer("face pipeline"):
    with ctx.Pool(os.cpu_count(), initializer=AF._init_worker, initargs=(model_path, CROP), maxtasksperchild=20) as pool:
        handles = [(j[0], pool.apply_async(AF.process_child, (j,))) for j in jobs]
        for k, (cid, h) in enumerate(handles):
            try:
                results.append(h.get(timeout=1800))
            except Exception as e:
                failed.append((cid, repr(e)[:200]))
            if k % 20 == 0:
                print(f"{k+1}/{len(jobs)} children", flush=True)
print("failed children:", failed or "none")
assert results, "face pipeline produced nothing"
index = pd.DataFrame([row for r in results for row in r["rows"]])
index = index.sort_values(["child_id", "clip_idx"]).reset_index(drop=True)
index.to_csv(OUT / "face_index.csv", index=False)
names = next((r["names"] for r in results if r.get("names")), None) or [f"bs_{i}" for i in range(AF.BLENDSHAPES)] + AF.EXTRA_NAMES
(OUT / "hand_feature_names.json").write_text(json.dumps(names))
print(index.head())

# %% [markdown]
# ## QC: detection rate per clip and per group
# If the face-detection rate differs between ASD and non-ASD children, the detector itself
# is a confound (it is reported, and NB09 tests whether it predicts the label).

# %%
index["valid_frac"] = index.n_valid / index.n_frames.clip(lower=1)
qc = index.merge(man[["child_id", "label", "sex", "age_band"]], on="child_id")
print(qc.groupby("label").valid_frac.describe())
qc["repeat_clip"] = qc.valid_frac < 0.5
print("clips to repeat (face visible < 50 %):", int(qc.repeat_clip.sum()))
qc.to_csv(OUT / "face_qc.csv", index=False)
dups = qc[qc.phash != "0"].groupby("phash").child_id.nunique()
print("identical first-crop hashes across children:", int((dups > 1).sum()))
sizes = {d.name: round(sum(f.stat().st_size for f in d.glob("*.npy")) / 1e9, 3) for d in OUT.iterdir() if d.is_dir()}
print("output GB:", sizes)
