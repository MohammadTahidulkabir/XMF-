# %% [meta]
# {"slug": "asd-xmf-d3-video-dataset", "title": "ASD-XMF D3 Video Dataset",
#  "gpu": false, "internet": true,
#  "datasets": ["gursewakbfcet/mmasd-final-dataset"], "kernels": [], "phase": "D"}

# %% [markdown]
# # D3 - XMF-Video: behaviour sequences (MMASD 2D skeleton)
# MMASD (IRB-approved, parental consent; 32 autistic children, therapy activities) ships privacy-preserving
# OpenPose skeletons instead of raw video.  Plan 4.2: pre-train the motion/behaviour branch.
# Each clip -> 64 frames x 25 joints x (x, y, confidence), centred on the neck and scaled by torso
# length, with a per-frame validity mask.  The subject ID is kept so that every split is by child.
# The eye-tracking (gaze) data are used in NB13 with the same child-wise protocol.

# %%
import json as _json, re
import numpy as np, pandas as pd
from pathlib import Path
import multiprocessing as mp
OUT = U.WORK / "xmf_video"; OUT.mkdir(parents=True, exist_ok=True)
T = 64
root = U.find_path("2D_openpose", want="dir")
clips = sorted({p.parent for p in root.rglob("*_keypoints.json")})
print("clips:", len(clips))
if PREFLIGHT:
    clips = clips[:60]

def load_clip(d):
    d = Path(d)
    frames = sorted(d.glob("*_keypoints.json"))
    seq = []
    for f in frames:
        try:
            ppl = _json.loads(f.read_text()).get("people", [])
        except Exception:
            ppl = []
        if ppl:  # most confident person in the frame
            k = max((np.array(p["pose_keypoints_2d"], np.float32).reshape(-1, 3) for p in ppl), key=lambda a: a[:, 2].sum())
            seq.append(k[:25])
        else:
            seq.append(np.zeros((25, 3), np.float32))
    seq = np.stack(seq) if seq else np.zeros((1, 25, 3), np.float32)
    valid = seq[:, :, 2].sum(1) > 0
    neck, hip = seq[:, 1, :2], seq[:, 8, :2]
    scale = np.linalg.norm(neck - hip, axis=1)
    scale = np.where(scale > 1, scale, np.nanmedian(scale[scale > 1]) if (scale > 1).any() else 1.0)
    xy = (seq[:, :, :2] - neck[:, None]) / scale[:, None, None]
    feat = np.concatenate([xy, seq[:, :, 2:3]], -1) * valid[:, None, None]
    ids = np.linspace(0, len(feat) - 1, T).round().astype(int)
    return feat[ids].reshape(T, -1).astype(np.float32), valid[ids], len(frames)

with mp.get_context("fork").Pool(os.cpu_count()) as pool:  # json parsing only: fork is safe here
    out = pool.map(load_clip, [str(c) for c in clips], chunksize=8)
X = np.stack([o[0] for o in out]); Mk = np.stack([o[1] for o in out])
meta = []
for c, o in zip(clips, out):
    m = re.match(r"^([a-z]+)_(\d+)_", c.name)
    meta.append({"clip": c.name, "activity": c.parent.name, "subject": m.group(2) if m else c.name.split("_")[1], "n_frames": o[2]})
meta = pd.DataFrame(meta)
keep = Mk.sum(1) >= T // 4
X, Mk, meta = X[keep], Mk[keep], meta[keep].reset_index(drop=True)
np.save(OUT / "skeleton.npy", X); np.save(OUT / "mask.npy", Mk)
meta.to_csv(OUT / "manifest.csv", index=False)
print(meta.activity.value_counts(), "\nsubjects:", meta.subject.nunique(), "| clips kept:", len(meta))
(OUT / "DATASHEET.md").write_text(f"""# XMF-Video
{len(meta)} MMASD skeleton clips, {meta.subject.nunique()} children, {meta.activity.nunique()} activities.
Tensor: {T} frames x 25 joints x (x, y, conf), neck-centred, torso-scaled; mask = frame has a detected person.
Source: MMASD (arXiv 2306.08243) via gursewakbfcet/mmasd-final-dataset. Autistic children only: activity labels, not diagnosis.
""")
