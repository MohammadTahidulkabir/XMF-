# %% [meta]
# {"slug": "asd-xmf-nb02-audio-pipeline", "title": "ASD-XMF NB02 Audio Pipeline",
#  "gpu": true, "internet": true,
#  "datasets": [], "kernels": ["asd-xmf-nb00-data-audit-manifest"], "phase": "B"}

# %% [markdown]
# # NB02 - Voice and cry pipeline for the cohort
# * **Voice**: the tablet microphone during each stimulus clip (`audio_path`, or extracted
#   from the clip video with ffmpeg).  Clips where the child did not vocalise
#   (voiced fraction < 5 %) are treated as *missing*, which is exactly the situation the
#   missing-modality fusion is designed for (pre-verbal children).
# * **Cry**: naturally occurring cries recorded in clinic (never provoked), `cry.csv`.
#
# Per item: log-mel `.npy`, eGeMAPS-v02 (openSMILE), wav2vec2-base layer means, AST
# embedding.  `voice_features.npy` / `cry_features.npy` (+ `_index.csv` with `child_id`,
# `clip_idx`) are what the fusion model reads.

# %%
import numpy as np, pandas as pd
from pathlib import Path
import asd_audio as A
U.pip_install("opensmile")
OUT = U.WORK / "audio"; OUT.mkdir(parents=True, exist_ok=True)
MIN_VOICED = 0.05
mdir = U.find_path("manifest.csv").parent
man = pd.read_csv(mdir / "manifest.csv")
clips = pd.read_csv(mdir / "clips.csv")
cry = pd.read_csv(mdir / "cry.csv")
root, is_dev = U.locate_cohort()
print("cohort:", root, "DEV" if is_dev else "REAL")
if PREFLIGHT:
    keep = man.child_id.head(8)
    clips, cry = clips[clips.child_id.isin(keep)], cry[cry.child_id.isin(keep)]

# %%
clips["audio_src"] = [a if isinstance(a, str) and a.strip() else v for a, v in zip(clips.get("audio_path", pd.Series([""] * len(clips))).fillna(""), clips.video_path)]
with U.Timer("voice"):
    vidx = A.extract_all(clips, "audio_src", root, OUT, "voice", use_ast=not PREFLIGHT, max_s=12)
with U.Timer("cry"):
    cidx = A.extract_all(cry, "cry_path", root, OUT, "cry", use_ast=not PREFLIGHT, max_s=12) if len(cry) else pd.DataFrame()

# %% [markdown]
# ## Fusion-ready vectors (drop non-vocal clips -> missing voice for that clip)

# %%
for tag, idx in (("voice", vidx), ("cry", cidx)):
    if not len(idx):
        continue
    X = A.fusion_audio_features(OUT, tag)
    ok = (idx.voiced_frac >= MIN_VOICED).values if tag == "voice" else np.ones(len(idx), bool)
    np.save(OUT / f"{tag}_features.npy", X[ok].astype(np.float32))
    idx[ok].reset_index(drop=True).to_csv(OUT / f"{tag}_fusion_index.csv", index=False)  # rows of *_features.npy
    print(tag, "items kept", int(ok.sum()), "/", len(idx), "| dim", X.shape[1])
    m = idx.merge(man[["child_id", "label"]], on="child_id")
    print(m.groupby("label").voiced_frac.describe()[["mean", "50%"]])
cov = pd.DataFrame({"child_id": man.child_id})
for tag in ("voice", "cry"):
    f = OUT / f"{tag}_fusion_index.csv"
    cov[f"has_{tag}"] = cov.child_id.isin(pd.read_csv(f).child_id) if f.exists() else False
print(cov.merge(man[["child_id", "label"]]).groupby("label")[["has_voice", "has_cry"]].mean())
cov.to_csv(OUT / "audio_coverage.csv", index=False)
