"""
asd_bench.py - per-sector model comparisons (photo, audio, video, text).

Every task trains ONE model under the same protocol and writes
  results/<name>.json  : train / val / test accuracy, macro-F1, balanced acc,
                         generalisation gap (train - val), best epoch, params, time
  curves/<name>.csv    : per-epoch train/val loss + accuracy (over/underfitting check)
  cm/<name>.npy        : confusion matrix on the test split
Early stopping on validation macro-F1, AdamW + cosine schedule, label smoothing,
mixed precision, channels_last.  The test split is used once, at the end.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import asd_utils as U
import asd_models as M


def cls_metrics(y, pred, n_cls) -> Dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    y, pred = np.asarray(y), np.asarray(pred)
    if len(y) == 0:
        return {"n": 0}
    return {"n": int(len(y)), "acc": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(n_cls)), zero_division=0)),
            "bal_acc": float(balanced_accuracy_score(y, pred)),
            "cm": confusion_matrix(y, pred, labels=list(range(n_cls))).tolist()}


def boot_ci(y, pred, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    vals = [fn(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _save(out: Path, name: str, res: Dict, curves: List[Dict]):
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "curves").mkdir(parents=True, exist_ok=True)
    (out / "results" / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
    pd.DataFrame(curves).to_csv(out / "curves" / f"{name}.csv", index=False)


# ----------------------------------------------------------------------------
# PHOTO: 7-class facial expression on the unified XMF-Photo dataset
# ----------------------------------------------------------------------------
PROPOSED_PHOTO = "xmf_face"   # plan 5.1: EfficientNet-B0 expression encoder with 256-d embedding


def build_photo_model(name: str, n_cls: int, img: int):
    if name == PROPOSED_PHOTO:
        return M.ExprNet("efficientnet_b0", n_cls=n_cls), img
    if name == "mobilenetv3_l":           # kept as ExprNet so the app can reuse its 256-d embedding
        return M.ExprNet("mobilenetv3_l", n_cls=n_cls), img
    return M.create_frame_model(name, img, n_out=n_cls)


def photo_task(job: Dict) -> None:
    out = Path(job["out_dir"])
    flag = out / "flags" / f"{job['name']}.done"
    if flag.exists():
        return
    U.seed_everything(job.get("seed", 0))
    dev = M.device()
    cache = Path(job["cache"])
    y = np.load(cache / "labels.npy")
    split = np.load(cache / "split.npy", allow_pickle=True)
    src = np.load(cache / "source.npy", allow_pickle=True)
    n_cls = int(job.get("n_cls", 7))
    tr, va, te = (np.flatnonzero(split == s) for s in ("train", "val", "test"))
    if job.get("limit"):
        rng = np.random.default_rng(0)
        tr = rng.choice(tr, min(len(tr), job["limit"]), replace=False)
        va, te = va[: job["limit"] // 4], te[: job["limit"] // 2]
    model, size = build_photo_model(job["model"], n_cls, job["img_size"])
    model = model.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=job["lr"], weight_decay=job.get("wd", 0.05))
    scaler = M.make_scaler()
    counts = np.bincount(y[tr], minlength=n_cls).astype(float)
    w = torch.tensor(np.clip((counts.sum() / np.maximum(counts, 1) / n_cls) ** 0.5, 0.3, 3), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    bs = job["batch"]
    mk = lambda ids, sh: torch.utils.data.DataLoader(M.MemmapImages(cache / "images.npy", y, ids), batch_size=bs if sh else bs * 2,
                                                      shuffle=sh, num_workers=job.get("workers", 2), pin_memory=True, drop_last=sh)
    steps_per = max(1, len(tr) // bs)
    total = job["epochs"] * steps_per
    ck = out / "ckpt" / f"{job['name']}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)

    def predict(ids):
        model.eval()
        P, L = [], 0.0
        with torch.no_grad(), M.amp_ctx():
            for xb, yb in mk(ids, False):
                lo = model(M.gpu_prepare(xb.to(dev, non_blocking=True), size, train=False)).float()
                L += F.cross_entropy(lo, yb.to(dev), reduction="sum").item()
                P.append(lo.argmax(1).cpu().numpy())
        P = np.concatenate(P) if P else np.zeros(0, int)
        return P, L / max(1, len(ids))

    curves, best, best_ep, step, t0 = [], -1.0, -1, 0, time.time()
    tr_eval = np.random.default_rng(1).choice(tr, min(len(tr), 6000), replace=False)
    for ep in range(job["epochs"]):
        model.train()
        run, n = 0.0, 0
        for xb, yb in mk(tr, True):
            xb = M.gpu_prepare(xb.to(dev, non_blocking=True), size, train=True, gray=True)
            yb = yb.to(dev)
            M.cosine_lr(opt, [job["lr"]], step, total, warmup=steps_per)
            lam, perm = 1.0, None
            if np.random.rand() < 0.5:   # mixup on half the batches (regularisation)
                lam = float(np.random.beta(0.4, 0.4))
                perm = torch.randperm(xb.shape[0], device=dev)
                xb = lam * xb + (1 - lam) * xb[perm]
            with M.amp_ctx():
                lo = model(xb).float()
                loss = lam * crit(lo, yb) + ((1 - lam) * crit(lo, yb[perm]) if perm is not None else 0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            run += loss.item() * len(yb)
            n += len(yb)
            step += 1
            if job.get("max_steps") and step >= job["max_steps"]:
                break
        ptr, ltr = predict(tr_eval)
        pva, lva = predict(va)
        mtr, mva = cls_metrics(y[tr_eval], ptr, n_cls), cls_metrics(y[va], pva, n_cls)
        curves.append({"epoch": ep, "train_loss": ltr, "val_loss": lva, "train_acc": mtr["acc"], "val_acc": mva["acc"],
                       "val_macro_f1": mva["macro_f1"], "sec": time.time() - t0})
        print(f"[{job['name']}] ep{ep} train_acc={mtr['acc']:.3f} val_acc={mva['acc']:.3f} val_f1={mva['macro_f1']:.3f}", flush=True)
        if mva["macro_f1"] > best:
            best, best_ep = mva["macro_f1"], ep
            torch.save(model.state_dict(), ck)
        elif ep - best_ep >= job.get("patience", 5):
            break
        if job.get("max_steps") and step >= job["max_steps"]:
            break
    model.load_state_dict(torch.load(ck, map_location=dev))
    ptr, _ = predict(tr_eval)
    pva, _ = predict(va)
    pte, _ = predict(te)
    res = {"model": job["model"], "proposed": job["model"] == PROPOSED_PHOTO, "params_m": M.count_params(model) / 1e6,
           "best_epoch": best_ep, "epochs_run": len(curves), "train_minutes": (time.time() - t0) / 60,
           "train": cls_metrics(y[tr_eval], ptr, n_cls), "val": cls_metrics(y[va], pva, n_cls),
           "test": cls_metrics(y[te], pte, n_cls)}
    res["gap_train_minus_val_acc"] = res["train"]["acc"] - res["val"]["acc"]
    res["test_acc_ci"] = boot_ci(y[te], pte, lambda a, b: float((a == b).mean()))
    res["test_by_source"] = {s: cls_metrics(y[te][src[te] == s], pte[src[te] == s], n_cls) for s in sorted(set(src[te]))}
    _save(out, job["name"], res, curves)
    np.save(out / "results" / f"{job['name']}_test_pred.npy", pte.astype(np.int8))
    if isinstance(model, M.ExprNet):  # weights reused downstream (NB04, NB05, NB12) under the NB03 names
        base = "efficientnet_b0" if job["model"] == PROPOSED_PHOTO else "mobilenetv3_l"
        (out / "weights").mkdir(exist_ok=True)
        torch.save(model.backbone.state_dict(), out / "weights" / f"{base}_expr_backbone.pt")
        torch.save({"emb_head": model.emb_head.state_dict(), "cls_head": model.cls_head.state_dict(),
                    "classes": M.EXPR_CLASSES, "img_size": size}, out / "weights" / f"{base}_expr_head.pt")
    ck.unlink(missing_ok=True)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("ok")


# ----------------------------------------------------------------------------
# VIDEO: skeleton-sequence behaviour recognition (MMASD), subject-wise folds
# ----------------------------------------------------------------------------
class SeqMLP(nn.Module):
    def __init__(self, d_in, n_cls, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d_in, h), nn.GELU(), nn.Dropout(0.3), nn.Linear(h, n_cls))

    def forward(self, x, m):
        mf = m.float().unsqueeze(-1)
        mu = (x * mf).sum(1) / mf.sum(1).clamp_min(1)
        sd = (((x - mu.unsqueeze(1)) ** 2) * mf).sum(1).div(mf.sum(1).clamp_min(1)).sqrt()
        return self.net(torch.cat([mu, sd], -1))


class SeqCNN(nn.Module):
    def __init__(self, d_in, n_cls, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(d_in, h, 5, padding=2), nn.BatchNorm1d(h), nn.GELU(),
                                 nn.Conv1d(h, h, 5, padding=2), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.3))
        self.fc = nn.Linear(h, n_cls)

    def forward(self, x, m):
        h = self.net(x.transpose(1, 2)).transpose(1, 2)
        mf = m.float().unsqueeze(-1)
        return self.fc((h * mf).sum(1) / mf.sum(1).clamp_min(1))


class SeqLSTM(nn.Module):
    """BiLSTM, or CNN-LSTM (the proposal's CNN + LSTM family) when cnn=True."""

    def __init__(self, d_in, n_cls, h=128, cnn=False):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(d_in, h, 5, padding=2), nn.BatchNorm1d(h), nn.GELU()) if cnn else None
        self.lstm = nn.LSTM(h if cnn else d_in, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * h, n_cls)

    def forward(self, x, m):
        if self.cnn is not None:
            x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(x)
        mf = m.float().unsqueeze(-1)
        return self.fc(self.drop((h * mf).sum(1) / mf.sum(1).clamp_min(1)))


class SeqXMF(nn.Module):
    """Proposed temporal encoder (plan 5.1): projection -> 2-layer transformer -> attention pooling."""

    def __init__(self, d_in, n_cls, d=128):
        super().__init__()
        self.enc = M.TemporalFace(d_in, d, layers=2, heads=4, dropout=0.2, max_t=256)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, n_cls))

    def forward(self, x, m):
        tok, _ = self.enc(x, m)
        return self.fc(tok)


SEQ_MODELS = {"mlp_pooled": SeqMLP, "cnn1d": SeqCNN, "bilstm": SeqLSTM,
              "cnn_lstm": lambda d, c: SeqLSTM(d, c, cnn=True), "xmf_temporal": SeqXMF}
PROPOSED_SEQ = "xmf_temporal"


def seq_cv(X, mask, y, groups, n_cls, models, out: Path, tag: str, epochs=80, folds=5, seed=0, max_steps=None) -> pd.DataFrame:
    """Subject-wise StratifiedGroupKFold for every sequence model (same folds for all)."""
    from sklearn.model_selection import StratifiedGroupKFold
    dev = M.device()
    sgk = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    split = list(sgk.split(X, y, groups))
    rows = []
    for name in models:
        oof = np.full(len(y), -1)
        curves, trs, vas, t0 = [], [], [], time.time()
        for f, (tr_all, te) in enumerate(split):
            U.seed_everything(seed + f)
            rng = np.random.default_rng(seed + f)
            g_tr = np.unique(groups[tr_all])
            g_va = rng.choice(g_tr, max(1, len(g_tr) // 6), replace=False)
            va = tr_all[np.isin(groups[tr_all], g_va)]
            tr = tr_all[~np.isin(groups[tr_all], g_va)]
            mu, sd = X[tr][mask[tr]].mean(0), X[tr][mask[tr]].std(0) + 1e-5
            Xn = torch.tensor(((X - mu) / sd) * mask[..., None], dtype=torch.float32, device=dev)
            Mk = torch.tensor(mask, device=dev)
            Y = torch.tensor(y, device=dev)
            model = SEQ_MODELS[name](X.shape[-1], n_cls).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
            best, best_ep, best_state, step = -1, -1, None, 0
            for ep in range(epochs):
                model.train()
                order = tr[rng.permutation(len(tr))]
                for s in range(0, len(order), 64):
                    idx = torch.tensor(order[s:s + 64], device=dev)
                    xb = Xn[idx] + 0.05 * torch.randn_like(Xn[idx])  # jitter augmentation
                    loss = F.cross_entropy(model(xb, Mk[idx]), Y[idx], label_smoothing=0.1)
                    opt.zero_grad(); loss.backward(); opt.step(); step += 1
                    if max_steps and step >= max_steps:
                        break
                model.eval()
                with torch.no_grad():
                    ptr = model(Xn[tr], Mk[tr]).argmax(1).cpu().numpy()
                    pva = model(Xn[va], Mk[va]).argmax(1).cpu().numpy()
                mva = cls_metrics(y[va], pva, n_cls)
                curves.append({"fold": f, "epoch": ep, "train_acc": float((ptr == y[tr]).mean()), "val_acc": mva["acc"], "val_macro_f1": mva["macro_f1"]})
                if mva["macro_f1"] > best:
                    best, best_ep, best_state = mva["macro_f1"], ep, {k: v.clone() for k, v in model.state_dict().items()}
                    best_tr = float((ptr == y[tr]).mean())
                    best_va = mva["acc"]
                elif ep - best_ep >= 12:
                    break
                if max_steps and step >= max_steps:
                    break
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                oof[te] = model(Xn[te], Mk[te]).argmax(1).cpu().numpy()
            trs.append(best_tr); vas.append(best_va)
        m = cls_metrics(y, oof, n_cls)
        lo, hi = boot_ci(y, oof, lambda a, b: float((a == b).mean()))
        res = {"task": tag, "model": name, "proposed": name == PROPOSED_SEQ, "params_m": M.count_params(SEQ_MODELS[name](X.shape[-1], n_cls)) / 1e6,
               "train_acc": float(np.mean(trs)), "val_acc": float(np.mean(vas)), "gap_train_minus_val_acc": float(np.mean(trs) - np.mean(vas)),
               "test": m, "test_acc_ci": [lo, hi], "minutes": (time.time() - t0) / 60}
        _save(out, f"{tag}__{name}", res, curves)
        rows.append({"task": tag, "model": name, "proposed": name == PROPOSED_SEQ, "test_acc": m["acc"], "ci_lo": lo, "ci_hi": hi,
                     "test_macro_f1": m["macro_f1"], "train_acc": res["train_acc"], "val_acc": res["val_acc"],
                     "gap": res["gap_train_minus_val_acc"], "params_m": res["params_m"]})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def comparison_table(rows: pd.DataFrame, metric="test_acc") -> pd.DataFrame:
    """Sort with the proposed model on top of its sector and mark over/underfitting."""
    df = rows.copy()
    df["fit"] = np.where(df["gap"] > 0.10, "overfit?", np.where(df["train_acc"] < 0.6, "underfit?", "ok"))
    return df.sort_values(["task", "proposed", metric], ascending=[True, False, False])
