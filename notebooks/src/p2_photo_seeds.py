# %% [meta]
# {"slug": "asd-xmf-p2-photo-seeds", "title": "ASD-XMF P2 Photo Seeds",
#  "gpu": true, "internet": true, "datasets": [], "kernels": ["asd-xmf-r1-photo-data-gaze"], "phase": "R"}

# %% [markdown]
# # P2 - Photo sector over 5 seeds, and on a near-duplicate-free test set
# The single-seed P1 run put XMF-Face v2 (0.766) about one point above the next models (ConvNeXt-T 0.755,
# Swin-T 0.755, EfficientNet-B0 0.748).  Run-to-run noise of about +-1.5 points is the same size, so P2 repeats
# the proposed model and its three closest competitors with 5 seeds (0-4), each with exactly the P1 recipe:
#
# * XMF-Face v2 (EfficientNet-B0 + distillation from an EfficientNetV2-S teacher, EMA, flip TTA); the
#   teacher is re-trained for every seed, so the teacher's own noise is part of the spread
# * EfficientNet-B0 (same backbone, plain recipe), ConvNeXt-T, Swin-T
#
# Reported: mean +- SD over seeds, per-seed wins, paired t-test over seeds, McNemar per seed (Holm), and
# accuracy on the **clean** test set without near-copies of training images (pHash distance <= 4 bits;
# <= 6 bits as a sensitivity check; from section `dup`).

# %%
import numpy as np, pandas as pd
from scipy import stats
import asd_bench as B
OUT = U.WORK / "p2"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = U.find_path("images.npy").parent
SEEDS = [0, 1] if PREFLIGHT else [0, 1, 2, 3, 4]
BASE = ["efficientnet_b0", "convnext_tiny", "swin_t"]
HEAVY = {"convnext_tiny", "swin_t", "efficientnetv2_s"}
print("photo cache:", CACHE, "| seeds:", SEEDS)

def job_for(m, name, seed, **kw):
    return {"name": name, "model": m, "cache": str(CACHE), "out_dir": str(OUT), "img_size": 160,
            "epochs": 1 if PREFLIGHT else 20, "patience": 5, "batch": 16 if PREFLIGHT else (128 if m in HEAVY else 256),
            "lr": (5e-4 if m in ("swin_t", "convnext_tiny") else 1.5e-3), "seed": seed, "workers": 2,
            "limit": 600 if PREFLIGHT else None, "max_steps": 4 if PREFLIGHT else None, **kw}

with U.Timer("photo seeds"):
    # 1) baselines and one distillation teacher per seed (longest jobs first keeps both GPUs busy)
    step1 = [job_for("efficientnetv2_s", f"kd_teacher_s{s}", s, save_train_logits=True) for s in SEEDS] + \
            [job_for(m, f"{m}_s{s}", s) for m in ("swin_t", "convnext_tiny", "efficientnet_b0") for s in SEEDS]
    U.run_jobs_parallel("asd_bench", "photo_task", step1, log_dir=OUT / "logs")
    # 2) XMF-Face v2 per seed, distilled from that seed's teacher
    U.run_jobs_parallel("asd_bench", "photo_task",
                        [job_for("xmf_face", f"xmf_face_s{s}", s, teacher=str(OUT / "results" / f"kd_teacher_s{s}_train_logits.npy"),
                                 kd_t=4.0, kd_alpha=0.5, ema=0.998, tta=True, recipe="xmf_v2",
                                 epochs=1 if PREFLIGHT else 30, patience=8) for s in SEEDS], log_dir=OUT / "logs")

# %% [markdown]
# ## Per-seed results, full and clean test set

# %%
y = np.load(CACHE / "labels.npy"); split = np.load(CACHE / "split.npy", allow_pickle=True)
src = np.load(CACHE / "source.npy", allow_pickle=True)
te = np.flatnonzero(split == "test")
dup_p = U.find_path("near_dup_test.csv")
dup = pd.read_csv(dup_p) if dup_p else None
CLEAN_MAX = 4   # pHash bits; see section dup (label agreement by distance)
MODELS = ["xmf_face"] + BASE
rows, preds = [], {}
for m in MODELS:
    for s in SEEDS:
        rp = OUT / "results" / f"{m}_s{s}.json"
        if not rp.exists():
            print("missing", rp.name); continue
        r = json.loads(rp.read_text())
        p = np.load(OUT / "results" / f"{m}_s{s}_test_pred.npy"); preds[(m, s)] = p
        yt = y[te][: len(p)]
        row = {"model": m, "seed": s, "proposed": m == "xmf_face", "test_acc": float((p == yt).mean()),
               "test_macro_f1": r["test"]["macro_f1"], "val_acc": r["val"]["acc"], "train_acc": r["train"]["acc"],
               "best_epoch": r["best_epoch"], "minutes": r["train_minutes"]}
        for so in ("fer2013", "rafdb", "ckplus"):
            k = src[te][: len(p)] == so
            row[f"acc_{so}"] = float((p[k] == yt[k]).mean()) if k.any() else np.nan
        if dup is not None and len(dup) >= len(p):
            dist = dup.min_dist_trainval.values[: len(p)]
            clean = dist > CLEAN_MAX
            row["clean_test_acc"] = float((p[clean] == yt[clean]).mean())
            row["n_clean"] = int(clean.sum())
            row["near_dup_acc"] = float((p[~clean] == yt[~clean]).mean()) if (~clean).any() else np.nan
            c6 = dist > 6   # sensitivity: also drop look-alikes at 5-6 bits
            row["clean6_test_acc"] = float((p[c6] == yt[c6]).mean())
        rows.append(row)
per_seed = pd.DataFrame(rows)
per_seed.to_csv(OUT / "photo_seeds_per_seed.csv", index=False)
print(U.table_md(per_seed))

# %% [markdown]
# ## Mean +- SD over seeds, paired comparisons

# %%
metrics = [c for c in ("test_acc", "test_macro_f1", "clean_test_acc", "clean6_test_acc", "acc_fer2013", "acc_rafdb", "acc_ckplus") if c in per_seed]
summ = per_seed.groupby("model")[metrics].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ = summ.reset_index().sort_values("test_acc_mean", ascending=False)
summ["n_seeds"] = summ.model.map(per_seed.groupby("model").size())
print(U.table_md(summ, ".4f"))
summ.to_csv(OUT / "photo_seeds_summary.csv", index=False)

pair = []
for b in BASE:
    for met in [c for c in ("test_acc", "clean_test_acc", "clean6_test_acc", "test_macro_f1") if c in per_seed]:
        a_ = per_seed[per_seed.model == "xmf_face"].set_index("seed")[met]
        b_ = per_seed[per_seed.model == b].set_index("seed")[met]
        common = a_.index.intersection(b_.index)
        d = (a_[common] - b_[common]).values
        t = stats.ttest_rel(a_[common], b_[common]) if len(common) > 1 else None
        pair.append({"baseline": b, "metric": met, "n_seeds": len(common), "mean_diff": d.mean(), "sd_diff": d.std(ddof=1) if len(d) > 1 else np.nan,
                     "xmf_wins": int((d > 0).sum()), "p_paired_t": float(t.pvalue) if t is not None else np.nan})
pair = pd.DataFrame(pair)
pair["p_holm"] = U.holm(pair.p_paired_t.fillna(1).values)
print(U.table_md(pair, ".4f"))
pair.to_csv(OUT / "photo_seeds_paired_tests.csv", index=False)

mc = []
for s in SEEDS:
    if ("xmf_face", s) not in preds:
        continue
    for b in BASE:
        if (b, s) in preds:
            pa, pb = preds[("xmf_face", s)], preds[(b, s)]
            mc.append({"seed": s, "baseline": b, **U.mcnemar_test(y[te][: len(pa)], pa, pb)})
mc = pd.DataFrame(mc)
if len(mc):
    mc["p_holm"] = U.holm(mc.p.values)
    print(U.table_md(mc, ".4f"))
    mc.to_csv(OUT / "photo_seeds_mcnemar.csv", index=False)
