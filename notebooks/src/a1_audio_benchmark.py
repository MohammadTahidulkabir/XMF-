# %% [meta]
# {"slug": "asd-xmf-a1-audio-benchmark", "title": "ASD-XMF A1 Audio Benchmark",
#  "gpu": true, "internet": true, "datasets": [], "kernels": ["asd-xmf-d2-audio-dataset"], "phase": "A"}

# %% [markdown]
# # A1 - Audio sector: proposed XMF-Audio vs baselines (voice and cry)
# **Proposed (plan 5.1 voice/cry branch):** XMF-Audio = gated fusion of three feature blocks
# (wav2vec2-base mid-layer mean, eGeMAPS prosody, MFCC-40 statistics): each block -> 128-d projection,
# a learned softmax gate weights the blocks, block dropout during training.
# **Baselines:** MFCC + SVM, eGeMAPS + LR, wav2vec2 + LR, AST + LR, CNN14 + LR, log-mel EfficientNet-B0 CNN,
# log-mel CNN-LSTM (proposal).
#
# Cross-validation never mixes a speaker / infant between train and test: leave-one-speaker-out for
# `vocal_affect`, 5-fold group CV by infant for `cry_reason`, stratified 5-fold for `cry_detect`.
# Every model uses the same folds.  Train vs validation accuracy is reported (over/underfitting).

# %%
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import asd_models as M, asd_bench as B
SRC = U.find_path("audio_index.csv").parent
OUT = U.WORK / "a1"; OUT.mkdir(parents=True, exist_ok=True)
idx = pd.read_csv(SRC / "audio_index.csv")
def feat(name):
    p = SRC / f"audio_{name}.npy"
    if not p.exists():
        return None
    X = np.load(p).astype(np.float32)
    return np.nan_to_num(X[:, 4:10].mean(1) if name == "w2v" else X)
FEATS = {k: feat(k) for k in ("mfcc", "egemaps", "w2v", "ast", "cnn14")}
FEATS = {k: v for k, v in FEATS.items() if v is not None}
print({k: v.shape for k, v in FEATS.items()})

def folds(task_rows):
    d = idx.loc[task_rows]
    y, g = d.label.values, d.group.values
    if d.task.iloc[0] == "vocal_affect":
        return [(task_rows[g != s], task_rows[g == s]) for s in sorted(set(g))]
    if d.task.iloc[0] == "cry_reason":
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
        return [(task_rows[a], task_rows[b]) for a, b in cv.split(task_rows, y, g)]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return [(task_rows[a], task_rows[b]) for a, b in cv.split(task_rows, y)]

# %% [markdown]
# ## Proposed XMF-Audio head

# %%
class XMFAudio(nn.Module):
    def __init__(self, dims, n_cls, d=128, p_drop=0.3):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(k, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(0.2)) for k in dims])
        self.gate = nn.Linear(d * len(dims), len(dims))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, n_cls))
        self.p_drop = p_drop

    def forward(self, xs):
        H = torch.stack([p(x) for p, x in zip(self.proj, xs)], 1)        # B, blocks, d
        g = self.gate(H.flatten(1))
        if self.training and self.p_drop:                                 # block dropout
            drop = torch.rand_like(g) < self.p_drop
            drop[drop.all(1)] = False
            g = g.masked_fill(drop, -1e4)
        return self.head((g.softmax(-1).unsqueeze(-1) * H).sum(1))

def fit_xmf(blocks, y, tr, te, n_cls, seed=0, epochs=200):
    dev = M.device(); U.seed_everything(seed)
    rng = np.random.default_rng(seed)
    va = rng.choice(tr, max(2, len(tr) // 7), replace=False); fit = np.setdiff1d(tr, va)
    mus = [(b[fit].mean(0), b[fit].std(0) + 1e-5) for b in blocks]
    T = [torch.tensor((b - m) / s, dtype=torch.float32, device=dev) for b, (m, s) in zip(blocks, mus)]
    Y = torch.tensor(y, device=dev)
    w = torch.tensor(np.bincount(y[fit], minlength=n_cls).clip(1) ** -0.5, dtype=torch.float32, device=dev)
    net = XMFAudio([b.shape[1] for b in blocks], n_cls).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    best, best_state, best_ep = -1, None, 0
    for ep in range(epochs):
        net.train()
        for s in range(0, len(fit), 128):
            i = torch.tensor(fit[rng.permutation(len(fit))][s:s + 128], device=dev)
            loss = F.cross_entropy(net([t[i] for t in T]), Y[i], weight=w / w.mean(), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net([t[va] for t in T]).argmax(1).cpu().numpy()
        f1 = B.cls_metrics(y[va], pv, n_cls)["macro_f1"]
        if f1 > best:
            best, best_ep, best_state = f1, ep, {k: v.clone() for k, v in net.state_dict().items()}
        elif ep - best_ep >= 25:
            break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        p = lambda ids: net([t[ids] for t in T]).argmax(1).cpu().numpy()
        return p(te), (p(fit) == y[fit]).mean(), (p(va) == y[va]).mean()

# %% [markdown]
# ## Run every model on the same folds

# %%
rows = []
for task, g in idx.groupby("task"):
    rows_t = g.index.to_numpy()
    classes = sorted(g.label.unique()); cmap = {c: i for i, c in enumerate(classes)}
    y = idx.label.map(cmap).fillna(-1).astype(int).values
    n_cls = len(classes)
    F_ = folds(rows_t)
    if PREFLIGHT:
        F_ = F_[:2]
    models = {f"{k}_svm" if k == "mfcc" else f"{k}_lr": k for k in FEATS}
    preds = {m: np.full(len(idx), -1) for m in list(models) + ["xmf_audio"]}
    tr_acc = {m: [] for m in preds}
    va_acc = {m: [] for m in preds}
    for tr, te in F_:
        for m, k in models.items():
            X = FEATS[k]
            clf = (make_pipeline(StandardScaler(), SVC(C=3.0, class_weight="balanced")) if m.endswith("_svm")
                   else make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")))
            clf.fit(X[tr], y[tr])
            preds[m][te] = clf.predict(X[te]); tr_acc[m].append((clf.predict(X[tr]) == y[tr]).mean()); va_acc[m].append(np.nan)
        blocks = [FEATS[k] for k in ("w2v", "egemaps", "mfcc") if k in FEATS]
        pt, a_tr, a_va = fit_xmf(blocks, y, tr, te, n_cls, epochs=3 if PREFLIGHT else 200)
        preds["xmf_audio"][te] = pt; tr_acc["xmf_audio"].append(a_tr); va_acc["xmf_audio"].append(a_va)
    # log-mel CNN and CNN-LSTM on GPU, same folds
    lab = OUT / f"{task}_items.csv"; idx.assign(label_id=y).to_csv(lab, index=False)
    jobs = [{"name": f"{arch}_{task}_f{f}", "arch": arch, "index": str(lab), "mel_dir": str(SRC / "audio_mel"), "label_col": "label_id",
             "train_rows": tr.tolist(), "test_rows": te.tolist(), "fold": f, "out_dir": str(OUT), "seed": 0,
             "epochs": 1 if PREFLIGHT else 30, "batch": 64, "lr": 1e-3, "max_steps": 3 if PREFLIGHT else None, "crop_frames": 300}
            for arch in ("effb0", "cnn_lstm") for f, (tr, te) in enumerate(F_)]
    U.run_jobs_parallel("asd_audio", "melcnn_fold_task", jobs, log_dir=OUT / "logs")
    for arch, name in (("effb0", "logmel_cnn_effb0"), ("cnn_lstm", "logmel_cnn_lstm")):
        preds[name] = np.full(len(idx), -1); tr_acc[name] = []; va_acc[name] = []
        for f in range(len(F_)):
            d = pd.read_csv(OUT / "preds" / f"{arch}_{task}_f{f}.csv")
            pc = [c for c in d.columns if c.startswith("p_")]
            preds[name][d.row.values] = np.array([int(float(c[2:])) for c in pc])[d[pc].values.argmax(1)]
            tr_acc[name].append(d.train_acc.iloc[0]); va_acc[name].append(np.nan)
    for m, p in preds.items():
        sel = p[rows_t] >= 0
        mm = B.cls_metrics(y[rows_t][sel], p[rows_t][sel], n_cls)
        lo, hi = B.boot_ci(y[rows_t][sel], p[rows_t][sel], lambda a, b: float((a == b).mean()))
        rows.append({"task": task, "model": m, "proposed": m == "xmf_audio", "n": mm["n"], "test_acc": mm["acc"], "ci_lo": lo, "ci_hi": hi,
                     "test_macro_f1": mm["macro_f1"], "test_bal_acc": mm["bal_acc"], "train_acc": float(np.nanmean(tr_acc[m])),
                     "val_acc": float(np.nanmean(va_acc[m])) if np.isfinite(va_acc[m]).any() else np.nan,
                     "gap": float(np.nanmean(tr_acc[m]) - mm["acc"]), "majority_baseline": float(pd.Series(y[rows_t]).value_counts(normalize=True).iloc[0])})
        B._save(OUT, f"{task}__{m}", rows[-1] | {"cm": mm["cm"], "classes": classes}, [])

# %%
tab = B.comparison_table(pd.DataFrame(rows))
print(tab.to_markdown(index=False, floatfmt=".3f"))
tab.to_csv(OUT / "audio_comparison.csv", index=False)
