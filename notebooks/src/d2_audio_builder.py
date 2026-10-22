# %% [meta]
# {"slug": "asd-xmf-d2-audio-dataset", "title": "ASD-XMF D2 Audio Dataset",
#  "gpu": true, "internet": true,
#  "datasets": ["tahidulkabir/asd-xmf-recanvo", "warcoder/infant-cry-audio-corpus", "sanmithasadhish/infant-cry-dataset"],
#  "kernels": [], "phase": "D"}

# %% [markdown]
# # D2 - XMF-Audio: unified voice + cry dataset with cached features
# | task | source | label | grouping for CV |
# |---|---|---|---|
# | `vocal_affect` | ReCANVo (8 minimally verbal autistic people) | caregiver affect label (classes with >= 100 clips) | speaker |
# | `cry_reason` | Donate-a-Cry | belly pain / burping / discomfort / hungry / tired | infant ID |
# | `cry_detect` | infant-cry dataset | cry vs. non-cry | none available (stated as a limitation) |
#
# Exact-duplicate audio (MD5) is removed across all sources.  Each file is decoded once (16 kHz mono)
# and cached as: 64-band log-mel, MFCC-40 statistics (proposal), eGeMAPS-v02 (openSMILE),
# wav2vec2-base layer means, AST and PANNs-CNN14 embeddings.

# %%
import hashlib, re
import numpy as np, pandas as pd
from pathlib import Path
import asd_audio as A
for pkg in ("opensmile", "panns-inference"):
    U.pip_install(pkg)
OUT = U.WORK / "xmf_audio"; OUT.mkdir(parents=True, exist_ok=True)

# %%
items = []
meta = U.find_path("dataset_file_directory.csv"); rc_root = meta.parent
wav = {p.name: p for p in rc_root.rglob("*.wav")}
rc = pd.read_csv(meta)
keep = rc.Label.value_counts(); keep = keep[keep >= 100].index
for r in rc[rc.Label.isin(keep)].itertuples():
    if r.Filename in wav:
        items.append({"path": str(wav[r.Filename]), "task": "vocal_affect", "label": r.Label, "group": r.Participant, "source": "recanvo"})
cry_dir = U.find_path("donateacry_corpus", want="dir")
for cdir in (p for p in cry_dir.iterdir() if p.is_dir()):
    for f in cdir.glob("*.wav"):
        m = re.match(r"^([0-9A-Fa-f-]{36})", f.name)
        items.append({"path": str(f), "task": "cry_reason", "label": cdir.name, "group": m.group(1).lower() if m else f.stem[:8], "source": "donateacry"})
det = U.find_path("cry", want="dir")
for cdir in (p for p in det.parent.iterdir() if p.is_dir()):
    for f in cdir.glob("*.wav"):
        items.append({"path": str(f), "task": "cry_detect", "label": "cry" if cdir.name.lower() == "cry" else "non_cry",
                      "group": f.stem, "source": f"infantcry/{cdir.name}"})
items = pd.DataFrame(items)
items["md5"] = [hashlib.md5(Path(p).read_bytes()).hexdigest() for p in items.path]
n0 = len(items)
items = items.drop_duplicates("md5").reset_index(drop=True)
print(f"removed {n0 - len(items)} exact duplicates")
print(items.groupby(["task", "label"]).size())
if PREFLIGHT:
    items = items.groupby(["task", "label"]).head(4).reset_index(drop=True)

# %%
items["relpath"] = items.path
with U.Timer("features"):
    A.extract_all(items, "relpath", Path("/"), OUT, "audio", use_cnn14=not PREFLIGHT, max_s=10)
idx = pd.read_csv(OUT / "audio_index.csv")
print(idx.groupby("task")[["dur_s", "voiced_frac"]].mean())
(OUT / "DATASHEET.md").write_text(f"""# XMF-Audio
{len(idx)} clips after MD5 de-duplication. Tasks: {idx.task.value_counts().to_dict()}.
Sources: ReCANVo (Zenodo 5786860, CC BY 3.0 US), Donate-a-Cry (warcoder/infant-cry-audio-corpus), infant-cry dataset (sanmithasadhish/infant-cry-dataset).
Features per clip: log-mel (audio_mel/), MFCC-40 stats, eGeMAPS-v02, wav2vec2-base layer means, AST, CNN14.
Not an autism-diagnosis dataset: used to pre-train the voice and cry branches.
""")
