# %% [meta]
# {"slug": "asd-xmf-d1-photo-dataset", "title": "ASD-XMF D1 Photo Dataset",
#  "gpu": false, "internet": true,
#  "datasets": ["msambare/fer2013", "shuvoalok/raf-db-dataset", "davilsena/ckdataset"], "kernels": [], "phase": "D"}

# %% [markdown]
# # D1 - XMF-Photo: unified facial-expression image dataset
# Combines FER2013, RAF-DB and CK+ into one dataset with one 7-class label space
# (neutral, happy, sad, surprise, fear, disgust, anger) for pre-training the face branch (plan 4.2 / 5.1).
#
# * decoded once to 160 px RGB uint8 (`images.npy`), one row per image in `manifest.csv`
# * official FER2013 / RAF-DB test splits are kept as test; 10 % of train becomes validation
# * **CK+ is an external test set only** (its frames come in per-person sequences without subject IDs,
#   so a random split would leak)
# * perceptual-hash de-duplication across all sources: a training image that nearly duplicates a
#   test image is removed from training
# * the scraped "Autistic Children Facial Data Set" and its mirrors are excluded (plan section 1)

# %%
import numpy as np, pandas as pd, cv2
from pathlib import Path
import multiprocessing as mp
from functools import partial
import asd_models as M
IMG = 160
OUT = U.WORK / "xmf_photo"; OUT.mkdir(parents=True, exist_ok=True)
CLS = M.EXPR_CLASSES
FER = {"neutral": "neutral", "happy": "happy", "sad": "sad", "surprise": "surprise", "fear": "fear", "disgust": "disgust", "angry": "anger"}
RAF = {"1": "surprise", "2": "fear", "3": "disgust", "4": "happy", "5": "sad", "6": "anger", "7": "neutral"}
CKP = {0: "anger", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}  # 7 = contempt -> dropped

rows = []
def add_folder(root, mapping, source):
    for split in ("train", "test"):
        for cdir in (Path(root) / split).iterdir() if (Path(root) / split).exists() else []:
            lab = mapping.get(cdir.name.lower())
            if lab and cdir.is_dir():
                rows.extend({"path": str(f), "label": CLS.index(lab), "source": source, "orig_split": split}
                            for f in cdir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
add_folder(U.find_path("angry", want="dir").parent.parent, FER, "fer2013")
add_folder(U.find_path("DATASET", want="dir"), RAF, "rafdb")
ck = pd.read_csv(U.find_path("ckextended.csv"))
ck = ck[ck.emotion.isin(CKP)]
df = pd.DataFrame(rows)
print(df.groupby(["source", "orig_split"]).size(), "\nCK+ images:", len(ck))

# %% [markdown]
# ## Decode once (all CPU cores)

# %%
if PREFLIGHT:
    df = pd.concat([g.head(300) for _, g in df.groupby(["source", "orig_split"])]).reset_index(drop=True)
    ck = ck.head(100)
with mp.get_context("spawn").Pool(os.cpu_count()) as pool:
    imgs = list(pool.imap(partial(U.load_image_rgb, size=IMG), df.path.tolist(), chunksize=256))
ck_imgs = [cv2.cvtColor(cv2.resize(np.array(p.split(), np.uint8).reshape(48, 48), (IMG, IMG), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
           for p in ck.pixels]
ckdf = pd.DataFrame({"path": [f"ckplus/{i}" for i in ck.index], "label": [CLS.index(CKP[e]) for e in ck.emotion],
                     "source": "ckplus", "orig_split": "test"})
df = pd.concat([df, ckdf], ignore_index=True)
X = np.stack(imgs + ck_imgs)
print("decoded", X.shape)

# %% [markdown]
# ## Splits + cross-source pHash de-duplication

# %%
rng = np.random.default_rng(0)
df["split"] = np.where(df.orig_split == "test", "test", "train")
tr = df.index[df.split == "train"].to_numpy()
df.loc[rng.choice(tr, int(0.10 * len(tr)), replace=False), "split"] = "val"
df["phash"] = [U.phash(im) for im in X]
dups = U.near_duplicates(df.phash.tolist(), df.index.tolist(), max_dist=2)
drop = set()
for a, b in zip(dups.a, dups.b):
    sa, sb = df.split[a], df.split[b]
    if sa == sb == "train":
        drop.add(b)                    # duplicate inside train: keep one
    elif "test" in (sa, sb) and sa != sb:
        drop.add(a if sa != "test" else b)  # never train/validate on a near copy of a test image
    elif "val" in (sa, sb) and "train" in (sa, sb):
        drop.add(a if sa == "train" else b)
keep = np.array(sorted(set(df.index) - drop))
print(f"near-duplicate pairs: {len(dups)} | removed: {len(drop)}")
df, X = df.loc[keep].reset_index(drop=True), X[keep]

# %%
np.save(OUT / "images.npy", X)
np.save(OUT / "labels.npy", df.label.values.astype(np.int64))
np.save(OUT / "split.npy", df.split.values.astype(object))
np.save(OUT / "source.npy", df.source.values.astype(object))
df.drop(columns=["path"]).assign(label_name=df.label.map(dict(enumerate(CLS)))).to_csv(OUT / "manifest.csv", index=False)
summary = df.groupby(["source", "split"]).size().unstack(fill_value=0)
print(summary)
(OUT / "DATASHEET.md").write_text(f"""# XMF-Photo
Unified 7-class facial-expression dataset ({', '.join(CLS)}), {len(df)} images at {IMG} px RGB.
Sources: FER2013 (msambare/fer2013), RAF-DB (shuvoalok/raf-db-dataset), CK+ (davilsena/ckdataset; external test only).
Splits: official test sets kept; 10 % of training as validation; pHash near-duplicates (Hamming <= 2) removed across splits.
Use: pre-training of the face-expression encoder. Not an autism dataset. Licences: as stated by each original source (research use).

{summary.to_markdown()}
""")
