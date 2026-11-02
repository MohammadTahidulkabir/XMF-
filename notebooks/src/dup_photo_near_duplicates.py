# %% [meta]
# {"slug": "asd-xmf-dup-photo-near-duplicates", "title": "ASD-XMF DUP Photo Near Duplicates",
#  "gpu": false, "internet": true, "datasets": [], "kernels": ["asd-xmf-d1-photo-dataset"], "phase": "R"}

# %% [markdown]
# # DUP - Near-duplicate audit of the XMF-Photo test sets
# D1 removes test images whose pHash is *identical* to a training image.  Re-encoded, resized or slightly
# cropped copies differ in a few pHash bits and survive that rule.  FER2013 and RAF-DB carry no person IDs,
# so a person-level split is impossible; the closest check is this exhaustive near-duplicate search.
#
# For every test image: the smallest Hamming distance (0-64 bits) between its 64-bit pHash and any train or
# validation image (exact, all pairs).  R2 re-scores every model on the "clean" test set without images
# within `CLEAN_MAX_DIST` bits of a training image, so the reader sees how much near-copies inflate accuracy.

# %%
import numpy as np, pandas as pd
OUT = U.WORK / "dup"; OUT.mkdir(parents=True, exist_ok=True)
SRC = U.WORK / "xmf_photo" if (U.WORK / "xmf_photo" / "manifest.csv").exists() else U.find_path("images.npy").parent
CLEAN_MAX_DIST = 8   # <= 8 of 64 bits: visually the same picture (pHash literature uses 6-10)
man = pd.read_csv(SRC / "manifest.csv", dtype={"phash": str})
split = np.load(SRC / "split.npy", allow_pickle=True)
assert len(man) == len(split), "manifest and split.npy differ"
h = np.array([int(x) for x in man.phash], dtype=np.uint64)
te = np.flatnonzero(split == "test")          # the same order photo_task uses for test predictions
ref = np.flatnonzero(split != "test")         # train + validation
print(f"test images: {len(te)}  | train+val images: {len(ref)}")

try:
    popcount = np.bitwise_count                # numpy >= 2.0
except AttributeError:
    _LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    popcount = lambda a: _LUT[a.view(np.uint8)].reshape(*a.shape, 8).sum(-1)

min_d = np.empty(len(te), dtype=np.int16); nn_row = np.empty(len(te), dtype=np.int64)
hr = h[ref]
for s in range(0, len(te), 256):
    d = popcount(h[te[s:s + 256], None] ^ hr[None, :]).astype(np.int16)
    j = d.argmin(1)
    min_d[s:s + 256] = d[np.arange(len(j)), j]; nn_row[s:s + 256] = ref[j]
res = pd.DataFrame({"test_pos": np.arange(len(te)), "row": te, "source": man.source.values[te],
                    "label": man.label.values[te], "min_dist_trainval": min_d, "nearest_row": nn_row,
                    "nearest_same_label": man.label.values[nn_row] == man.label.values[te]})
res.to_csv(OUT / "near_dup_test.csv", index=False)

# %%
rows = []
for src, g in [("all", res)] + list(res.groupby("source")):
    r = {"source": src, "n_test": len(g)}
    for t in (0, 2, 4, 6, 8, 10, 12):
        r[f"le_{t}"] = int((g.min_dist_trainval <= t).sum())
    r["pct_le_8"] = round(100 * (g.min_dist_trainval <= CLEAN_MAX_DIST).mean(), 2)
    r["same_label_among_le_8"] = round(float(g.loc[g.min_dist_trainval <= CLEAN_MAX_DIST, "nearest_same_label"].mean()), 3) \
        if (g.min_dist_trainval <= CLEAN_MAX_DIST).any() else np.nan
    rows.append(r)
summ = pd.DataFrame(rows)
print(U.table_md(summ))
summ.to_csv(OUT / "near_dup_summary.csv", index=False)
U.save_json({"clean_max_dist": CLEAN_MAX_DIST, "n_test": len(te),
             "n_near_dup": int((min_d <= CLEAN_MAX_DIST).sum())}, OUT / "near_dup_config.json")
