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
# <= 4 of 64 bits: in the first full run ~85 % of test images at distance 2-4 shared the label of their nearest
# training image (true copies), at 6 bits 70 %, at 8 bits 46 % (near the ~25 % of unrelated faces) -
# small 48-px faces look alike to pHash, so wider thresholds catch look-alikes, not copies.  6 = sensitivity check.
CLEAN_MAX_DIST = 4
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
    r[f"pct_le_{CLEAN_MAX_DIST}"] = round(100 * (g.min_dist_trainval <= CLEAN_MAX_DIST).mean(), 2)
    rows.append(r)
summ = pd.DataFrame(rows)
print(U.table_md(summ))
summ.to_csv(OUT / "near_dup_summary.csv", index=False)
# how often a test image and its nearest training image share the expression label, by distance
# (true copies mostly share it; unrelated faces only by chance) - this is what sets CLEAN_MAX_DIST
by_d = res.groupby("min_dist_trainval").agg(n=("row", "size"), same_label=("nearest_same_label", "mean")).reset_index()
chance = float((man.label.value_counts(normalize=True) ** 2).sum())
by_d["chance_same_label"] = round(chance, 3)
print(U.table_md(by_d[by_d.min_dist_trainval <= 14]))
by_d.to_csv(OUT / "near_dup_label_agreement.csv", index=False)
U.save_json({"clean_max_dist": CLEAN_MAX_DIST, "n_test": len(te),
             "n_near_dup": int((min_d <= CLEAN_MAX_DIST).sum())}, OUT / "near_dup_config.json")
