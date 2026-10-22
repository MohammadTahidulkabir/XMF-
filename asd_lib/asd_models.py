"""
asd_models.py - models and GPU training tasks for BanglaASD-XMF.

  * GPU augmentation (uint8 in RAM/memmap -> augmented fp16 on GPU)
  * frame-level model zoo (timm CNNs / ViTs / Swin, YOLOv8-cls, YOLOv8-ATT)
  * frame_fold_task      : one fold of a frame-level baseline (NB04)
  * video_fold_task      : one fold of a video baseline (NB06)
  * XMF                  : proposed missing-modality-robust fusion network
  * fusion baselines     : early concat, late average, TFN, LMF, MulT-lite (+ stacking in sklearn)
  * fusion_fold_task     : one fold x seed of any fusion model / ablation (NB08, NB09)
  * evidential loss, MC-dropout, modality Shapley values (16 subsets)

All tasks are launched through asd_utils.run_jobs_parallel so that two folds
train side by side on the two T4s.  Each task writes preds CSVs + a done
flag and returns immediately if the flag exists (resume after timeouts).
"""
from __future__ import annotations

import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import asd_utils as U

MODALITIES = ("face", "voice", "cry", "quest")
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_ctx():
    return torch.autocast("cuda", dtype=torch.float16) if torch.cuda.is_available() else torch.autocast("cpu", enabled=False)


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    except Exception:  # older torch
        return torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def cosine_lr(opt, base_lrs, step, total, warmup):
    if step < warmup:
        f = (step + 1) / warmup
    else:
        f = 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    for g, lr in zip(opt.param_groups, base_lrs):
        g["lr"] = lr * f


# ----------------------------------------------------------------------------
# GPU augmentation
# ----------------------------------------------------------------------------
def gpu_prepare(x_uint8: torch.Tensor, size: int, train: bool, gray: bool = False) -> torch.Tensor:
    """x: [B,H,W,3] or [B,H,W] uint8 on GPU -> normalised [B,3,size,size],
    channels_last.  Augment: random resized crop (0.8-1), flip, brightness/
    contrast jitter, small rotation."""
    if x_uint8.dim() == 3:
        x_uint8 = x_uint8.unsqueeze(-1)
    x = x_uint8.permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    B = x.shape[0]
    if train:
        scale = torch.empty(B, device=x.device).uniform_(0.8, 1.0)
        ang = torch.empty(B, device=x.device).uniform_(-0.17, 0.17)
        tx = torch.empty(B, device=x.device).uniform_(-0.06, 0.06)
        ty = torch.empty(B, device=x.device).uniform_(-0.06, 0.06)
        flip = (torch.rand(B, device=x.device) < 0.5).float() * 2 - 1
        cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
        theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
        grid = F.affine_grid(theta, (B, 3, size, size), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        b = torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.15, 0.15)
        c = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - m) * c + m + b).clamp_(0, 1)
        if gray:
            g = (torch.rand(B, 1, 1, 1, device=x.device) < 0.2).float()
            x = g * x.mean(1, keepdim=True).expand_as(x) + (1 - g) * x
    elif x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    x = (x - IMNET_MEAN.to(x.device)) / IMNET_STD.to(x.device)
    return x.contiguous(memory_format=torch.channels_last)


# ----------------------------------------------------------------------------
# frame-level model zoo
# ----------------------------------------------------------------------------
TIMM_ZOO = {
    "resnet18": "resnet18.a1_in1k",
    "resnet50": "resnet50.a1_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "mobilenetv3_l": "mobilenetv3_large_100.ra_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "vit_s16": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "deit_s": "deit_small_patch16_224.fb_in1k",
    "swin_t": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
}
TRANSFORMER_NAMES = {"vit_s16", "deit_s", "swin_t"}


class CBAM(nn.Module):
    """Channel + spatial attention (Woo et al. 2018) - the 'ATT' in YOLOv8-ATT."""

    def __init__(self, c, r=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, max(8, c // r)), nn.ReLU(inplace=True), nn.Linear(max(8, c // r), c))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        a = torch.sigmoid(self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3))))[:, :, None, None]
        x = x * a
        s = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * s


class YoloCls(nn.Module):
    def __init__(self, variant="yolov8s-cls.pt", attention=False, n_out=1):
        super().__init__()
        from ultralytics import YOLO
        seq = YOLO(variant).model.model
        self.backbone = nn.Sequential(*list(seq)[:-1])
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            c = self.backbone(torch.zeros(1, 3, 160, 160)).shape[1]
        self.att = CBAM(c) if attention else nn.Identity()
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(c, n_out)
        self.num_features = c

    def forward_features(self, x):
        return self.att(self.backbone(x))

    def forward_head(self, f, pre_logits=False):
        f = f.mean((2, 3))
        return f if pre_logits else self.fc(self.drop(f))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def create_frame_model(name: str, img_size: int = 160, n_out: int = 1, pretrained: bool = True,
                       init_weights: Optional[str] = None):
    """Returns (model, input_size)."""
    import timm
    if name in ("yolov8_cls", "yolov8_att"):
        return YoloCls(attention=(name == "yolov8_att"), n_out=n_out), img_size
    if name.endswith("_expr"):  # NB03 expression-pretrained backbones
        base = name[: -len("_expr")]
        m = timm.create_model(TIMM_ZOO[base], pretrained=False, num_classes=n_out)
        if init_weights and Path(init_weights).exists():
            sd = torch.load(init_weights, map_location="cpu")
            sd = {k: v for k, v in sd.items() if not k.startswith(("classifier", "head", "fc", "emb_head", "cls_head"))}
            missing, unexpected = m.load_state_dict(sd, strict=False)
            print(f"[init] {name}: loaded expression weights ({len(missing)} missing, {len(unexpected)} unexpected)")
        else:
            print(f"[init] {name}: expression weights not found -> ImageNet init")
            m = timm.create_model(TIMM_ZOO[base], pretrained=pretrained, num_classes=n_out)
        return m, img_size
    tid = TIMM_ZOO[name]
    if name in TRANSFORMER_NAMES:
        kw = {"img_size": img_size}
        if name == "swin_t":
            kw["window_size"] = 5 if img_size % 160 == 0 else 7
        try:
            return timm.create_model(tid, pretrained=pretrained, num_classes=n_out, **kw), img_size
        except Exception as e:
            print(f"[zoo] {name} at {img_size}px failed ({e}); using 224px")
            return timm.create_model(tid, pretrained=pretrained, num_classes=n_out), 224
    return timm.create_model(tid, pretrained=pretrained, num_classes=n_out), img_size


def backbone_param_groups(model, head_lr, backbone_lr):
    head_names = ("classifier", "head", "fc", "att")
    head, body = [], []
    for n, p in model.named_parameters():
        (head if n.split(".")[0] in head_names else body).append(p)
    return [{"params": body, "lr": backbone_lr}, {"params": head, "lr": head_lr}], head, body


# ----------------------------------------------------------------------------
# face crop store (written by NB01)
# ----------------------------------------------------------------------------
class FaceStore:
    """face/crops/<child>.npy uint8 [C,T,H,W,3]; face/valid/<child>.npy bool
    [C,T]; face/hand/<child>.npy float16 [C,T,Dh]; face_index.csv."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.index = pd.read_csv(self.root / "face_index.csv")
        self._cache = {}

    def crops(self, cid):
        if cid not in self._cache:
            self._cache[cid] = np.load(self.root / "crops" / f"{cid}.npy", mmap_mode="r")
        return self._cache[cid]

    def valid(self, cid):
        return np.load(self.root / "valid" / f"{cid}.npy")

    def hand(self, cid):
        p = self.root / "hand" / f"{cid}.npy"
        return np.load(p).astype(np.float32) if p.exists() else None

    def has(self, cid):
        return (self.root / "crops" / f"{cid}.npy").exists()


class FrameDataset(torch.utils.data.Dataset):
    """Samples up to k frames per clip per epoch (bounded epoch cost)."""

    def __init__(self, store: FaceStore, ids: Sequence[str], labels: Dict[str, int], k_per_clip: int,
                 train: bool, stride: int = 2, seed: int = 0):
        self.store, self.labels = store, labels
        self.items = []
        rng = np.random.default_rng(seed)
        for cid in ids:
            if not store.has(cid):
                continue
            v = store.valid(cid)
            for c in range(v.shape[0]):
                ts = np.flatnonzero(v[c])
                if len(ts) == 0:
                    continue
                if train:
                    ts = rng.choice(ts, size=min(k_per_clip, len(ts)), replace=False)
                else:
                    ts = ts[::stride]
                self.items += [(cid, c, int(t)) for t in ts]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cid, c, t = self.items[i]
        img = np.ascontiguousarray(self.store.crops(cid)[c, t])
        return torch.from_numpy(img), self.labels[cid], i


def predict_children(model, loader, items, size, dev) -> pd.DataFrame:
    model.eval()
    probs = np.zeros(len(items), dtype=np.float32)
    with torch.no_grad(), amp_ctx():
        for x, _, idx in loader:
            x = gpu_prepare(x.to(dev, non_blocking=True), size, train=False)
            probs[idx.numpy()] = torch.sigmoid(model(x).float().squeeze(1)).cpu().numpy()
    df = pd.DataFrame(items, columns=["child_id", "clip", "t"])
    df["prob"] = probs
    clip = df.groupby(["child_id", "clip"], as_index=False).prob.mean()
    child = clip.groupby("child_id", as_index=False).agg(prob=("prob", "mean"), uncertainty=("prob", "std"))
    return child


def frame_fold_task(job: Dict) -> None:
    """NB04: one model x seed x fold on face crops.  Child decision = mean of
    clip means of frame probabilities."""
    out = Path(job["out_dir"])
    flag = out / "flags" / f"{job['name']}.done"
    if flag.exists():
        print("already done", flag)
        return
    U.seed_everything(job["seed"])
    dev = device()
    children = pd.read_csv(job["manifest"])
    tr, va, te = U.fold_ids(children, job["seed"], job["fold"])
    U.assert_no_leak(tr, va, te)
    labels = dict(zip(children.child_id, children.label.astype(int)))
    store = FaceStore(job["face_root"])
    model, size = create_frame_model(job["model"], job["img_size"], init_weights=job.get("init_weights"))
    model = model.to(dev).to(memory_format=torch.channels_last)
    bs, nw = job["batch"], job.get("workers", 2)
    mk = lambda ids, train: FrameDataset(store, ids, labels, job["k_per_clip"], train, seed=job["seed"])
    va_ds, te_ds = mk(va, False), mk(te, False)
    dl = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=bs if sh else bs * 2, shuffle=sh, num_workers=nw,
                                                     pin_memory=True, drop_last=sh, persistent_workers=False)
    groups, head, body = backbone_param_groups(model, job["lr"], job["lr"] * job.get("backbone_mult", 0.1))
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = make_scaler()
    pos_w = torch.tensor([(len(tr) - sum(labels[c] for c in tr)) / max(1, sum(labels[c] for c in tr))], device=dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w.clamp(0.3, 3.0))
    best, best_ep, patience = -2.0, -1, job.get("patience", 4)
    ck = out / "ckpt" / f"{job['name']}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    epochs = job["epochs"]
    step, total = 0, None
    for ep in range(epochs):
        tr_ds = mk(tr, True)
        tr_dl = dl(tr_ds, True)
        if total is None:
            total = epochs * len(tr_dl)
        frozen = ep < job.get("frozen_epochs", 1)
        for p in body:
            p.requires_grad_(not frozen)
        model.train()
        t0, run = time.time(), 0.0
        for x, y, _ in tr_dl:
            x = gpu_prepare(x.to(dev, non_blocking=True), size, train=True)
            y = y.float().to(dev)
            cosine_lr(opt, base_lrs, step, total, warmup=max(1, len(tr_dl) // 2))
            with amp_ctx():
                loss = crit(model(x).squeeze(1).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            run += loss.item()
            step += 1
            if job.get("max_steps") and step >= job["max_steps"]:
                break
        vdf = predict_children(model, dl(va_ds, False), va_ds.items, size, dev)
        vdf["y_true"] = vdf.child_id.map(labels)
        auc = U._safe_auc(vdf.y_true, vdf.prob)
        auc = -1 if np.isnan(auc) else auc
        print(f"[{job['name']}] ep{ep} loss={run/max(1,len(tr_dl)):.4f} val_auc={auc:.4f} {time.time()-t0:.0f}s", flush=True)
        if auc > best:
            best, best_ep = auc, ep
            torch.save(model.state_dict(), ck)
        elif ep - best_ep >= patience:
            break
        if job.get("max_steps") and step >= job["max_steps"]:
            break
    model.load_state_dict(torch.load(ck, map_location=dev))
    for part, ds in (("val", va_ds), ("test", te_ds)):
        if len(ds) == 0:
            continue
        df = predict_children(model, dl(ds, False), ds.items, size, dev)
        df["y_true"] = df.child_id.map(labels)
        df["part"] = part
        U.save_preds(df, job["model_tag"], job["fold"], out / "preds", job["seed"])
    if not job.get("keep_ckpt", False):
        ck.unlink(missing_ok=True)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(json.dumps({"best_auc": best, "best_ep": best_ep}))


# ----------------------------------------------------------------------------
# frame embeddings (NB05)
# ----------------------------------------------------------------------------
@torch.no_grad()
def embed_store(model, store: FaceStore, ids, size, out_path: Path, dim: int, batch: int = 256,
                fwd=None) -> np.ndarray:
    """Embeds every frame of every clip -> float16 memmap [N_clips, T, dim]
    in face_index order; invalid frames are zeros."""
    dev = device()
    idx = store.index[store.index.child_id.isin(ids)].reset_index(drop=True)
    first = store.crops(idx.child_id.iloc[0])
    T = first.shape[1]
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(len(idx), T, dim))
    model.eval()
    row = 0
    for cid, g in idx.groupby("child_id", sort=False):
        crops = store.crops(cid)
        valid = store.valid(cid)
        for _, r in g.iterrows():
            c = int(r.clip_idx)
            ts = np.flatnonzero(valid[c])
            out = np.zeros((T, dim), np.float16)
            for s in range(0, len(ts), batch):
                tt = ts[s:s + batch]
                x = gpu_prepare(torch.from_numpy(np.ascontiguousarray(crops[c, tt])).to(dev), size, train=False)
                with amp_ctx():
                    e = fwd(x) if fwd else model(x)
                out[tt] = e.float().cpu().numpy().astype(np.float16)
            arr[row] = out
            row += 1
    arr.flush()
    idx.to_csv(str(out_path).replace(".npy", "_index.csv"), index=False)
    return arr


# ----------------------------------------------------------------------------
# video baselines (NB06)
# ----------------------------------------------------------------------------
class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, store: FaceStore, ids, labels, n_frames=16, train=True, seed=0):
        self.store, self.labels, self.n, self.train = store, labels, n_frames, train
        self.rng = np.random.default_rng(seed)
        self.items = []
        for cid in ids:
            if not store.has(cid):
                continue
            v = store.valid(cid)
            for c in range(v.shape[0]):
                if v[c].sum() >= 4:
                    self.items.append((cid, c))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cid, c = self.items[i]
        ts = np.flatnonzero(self.store.valid(cid)[c])
        if self.train and len(ts) > self.n:
            start = self.rng.integers(0, len(ts) - self.n + 1)
            sel = ts[start:start + self.n]
        else:
            sel = ts[np.linspace(0, len(ts) - 1, self.n).round().astype(int)]
        x = np.ascontiguousarray(self.store.crops(cid)[c, sel])  # [n,H,W,3]
        return torch.from_numpy(x), self.labels[cid], i


def gpu_prepare_video(x: torch.Tensor, size: int, train: bool, mean=IMNET_MEAN, std=IMNET_STD) -> torch.Tensor:
    """[B,T,H,W,3] uint8 -> [B,3,T,size,size] float (same aug for all frames of a clip)."""
    B, T = x.shape[:2]
    x = x.permute(0, 1, 4, 2, 3).float().div_(255.0)  # B,T,3,H,W
    if train:
        flip = torch.rand(B, device=x.device) < 0.5
        x[flip] = x[flip].flip(-1)
        b = torch.empty(B, 1, 1, 1, 1, device=x.device).uniform_(-0.12, 0.12)
        x = (x + b).clamp_(0, 1)
    x = F.interpolate(x.flatten(0, 1), size=(size, size), mode="bilinear", align_corners=False).view(B, T, 3, size, size)
    x = (x - mean.to(x.device).view(1, 1, 3, 1, 1)) / std.to(x.device).view(1, 1, 3, 1, 1)
    return x.permute(0, 2, 1, 3, 4).contiguous()  # B,3,T,H,W


class VideoNet(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.layout = "BCTHW"
        if name == "r2plus1d_18":
            from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
            self.net = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
            self.net.fc = nn.Linear(self.net.fc.in_features, 1)
            self.size, self.T = 112, 16
        elif name == "x3d_s":
            self.net = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=True)
            proj = self.net.blocks[-1].proj
            self.net.blocks[-1].proj = nn.Linear(proj.in_features, 1)
            self.net.blocks[-1].activation = None
            self.size, self.T = 160, 13
        elif name == "videomae_s":
            from transformers import VideoMAEForVideoClassification
            self.net = VideoMAEForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-small-finetuned-kinetics", num_labels=1, ignore_mismatched_sizes=True)
            self.size, self.T = 224, 16
            self.layout = "BTCHW"
        else:
            raise ValueError(name)

    def forward(self, x):  # x: B,3,T,H,W
        if self.layout == "BTCHW":
            return self.net(pixel_values=x.permute(0, 2, 1, 3, 4)).logits
        out = self.net(x)
        return out.view(out.shape[0], -1)[:, :1]


def video_fold_task(job: Dict) -> None:
    out = Path(job["out_dir"])
    flag = out / "flags" / f"{job['name']}.done"
    if flag.exists():
        return
    U.seed_everything(job["seed"])
    dev = device()
    children = pd.read_csv(job["manifest"])
    tr, va, te = U.fold_ids(children, job["seed"], job["fold"])
    labels = dict(zip(children.child_id, children.label.astype(int)))
    store = FaceStore(job["face_root"])
    model = VideoNet(job["model"]).to(dev)
    T, size = model.T, model.size
    mk = lambda ids, train: ClipDataset(store, ids, labels, T, train, job["seed"])
    tr_ds, va_ds, te_ds = mk(tr, True), mk(va, False), mk(te, False)
    dl = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=job["batch"], shuffle=sh, num_workers=2,
                                                     pin_memory=True, drop_last=sh and len(ds) > job["batch"])
    opt = torch.optim.AdamW(model.parameters(), lr=job["lr"], weight_decay=0.05)
    scaler = make_scaler()
    crit = nn.BCEWithLogitsLoss()
    total = job["epochs"] * max(1, len(dl(tr_ds, True)))
    step, best, best_ep = 0, -2.0, -1
    ck = out / "ckpt" / f"{job['name']}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)

    def predict(ds):
        model.eval()
        probs = np.zeros(len(ds), np.float32)
        with torch.no_grad(), amp_ctx():
            for x, _, idx in dl(ds, False):
                probs[idx.numpy()] = torch.sigmoid(model(gpu_prepare_video(x.to(dev), size, False)).float().squeeze(1)).cpu().numpy()
        d = pd.DataFrame(ds.items, columns=["child_id", "clip"])
        d["prob"] = probs
        return d.groupby("child_id", as_index=False).agg(prob=("prob", "mean"), uncertainty=("prob", "std"))

    for ep in range(job["epochs"]):
        model.train()
        for x, y, _ in dl(tr_ds, True):
            cosine_lr(opt, [job["lr"]], step, total, warmup=20)
            with amp_ctx():
                loss = crit(model(gpu_prepare_video(x.to(dev), size, True)).squeeze(1).float(), y.float().to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if job.get("max_steps") and step >= job["max_steps"]:
                break
        v = predict(va_ds)
        auc = U._safe_auc(v.child_id.map(labels), v.prob)
        auc = -1 if np.isnan(auc) else auc
        print(f"[{job['name']}] ep{ep} val_auc={auc:.4f}", flush=True)
        if auc > best:
            best, best_ep = auc, ep
            torch.save(model.state_dict(), ck)
        elif ep - best_ep >= job.get("patience", 3):
            break
        if job.get("max_steps") and step >= job["max_steps"]:
            break
    model.load_state_dict(torch.load(ck, map_location=dev))
    for part, ds in (("val", va_ds), ("test", te_ds)):
        if len(ds):
            d = predict(ds)
            d["y_true"] = d.child_id.map(labels)
            d["part"] = part
            U.save_preds(d, job["model_tag"], job["fold"], out / "preds", job["seed"])
    ck.unlink(missing_ok=True)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(json.dumps({"best_auc": best}))


# ----------------------------------------------------------------------------
# fusion bundle: everything a fusion model needs, per child, in RAM
# ----------------------------------------------------------------------------
def build_fusion_bundle(manifest: Path, face_emb: Optional[Path], face_root: Optional[Path],
                        audio_root: Optional[Path], max_clips: int = 10, max_cry: int = 3,
                        quest_cols: Optional[List[str]] = None, out_path: Optional[Path] = None) -> Dict:
    """Assemble aligned arrays (one row per child).  Missing modalities are
    zeros with mask=0, which the models replace with learned missing tokens.
      face  [N,C,T,Df+Dh]  frame mask [N,C,T]  clip mask face_mask [N,C]
      voice [N,C,Dv]       voice_mask [N,C]
      cry   [N,K,Dc]       cry_mask   [N,K]
      quest [N,Dq]         quest_mask [N]
    """
    ch = pd.read_csv(manifest)
    ids = ch.child_id.tolist()
    pos = {c: i for i, c in enumerate(ids)}
    N = len(ids)
    B: Dict = {"child_id": np.array(ids, dtype=str), "y": ch.label.astype(int).values}
    # face
    if face_emb is not None and Path(face_emb).exists():
        E = np.load(face_emb, mmap_mode="r")
        idx = pd.read_csv(str(face_emb).replace(".npy", "_index.csv"))
        T, De = E.shape[1], E.shape[2]
        Dh = 0
        if face_root is not None:
            st = FaceStore(face_root)
            h0 = st.hand(idx.child_id.iloc[0])
            Dh = 0 if h0 is None else h0.shape[-1]
        face = np.zeros((N, max_clips, T, De + Dh), np.float16)
        fmask = np.zeros((N, max_clips, T), bool)
        for cid, g in idx.groupby("child_id", sort=False):
            if cid not in pos:
                continue
            i = pos[cid]
            valid = st.valid(cid) if face_root is not None else None
            hand = st.hand(cid) if (face_root is not None and Dh) else None
            for row, r in zip(g.index, g.itertuples()):
                c = int(r.clip_idx)
                if c >= max_clips:
                    continue
                face[i, c, :, :De] = E[row]
                if hand is not None:
                    face[i, c, :, De:] = np.nan_to_num(hand[c])
                fmask[i, c] = valid[c] if valid is not None else np.abs(E[row]).sum(-1) > 0
        B["face"], B["face_frame_mask"] = face, fmask
        B["face_mask"] = fmask.sum(-1) >= 3
        B["face_dims"] = (De, Dh)
    # audio
    for mod in ("voice", "cry"):
        f = (Path(audio_root) / f"{mod}_features.npy") if audio_root else None
        if f is not None and f.exists():
            X = np.load(f)
            ai = pd.read_csv(Path(audio_root) / f"{mod}_fusion_index.csv")
            K = max_clips if mod == "voice" else max_cry
            arr = np.zeros((N, K, X.shape[1]), np.float32)
            m = np.zeros((N, K), bool)
            for cid, g in ai.groupby("child_id", sort=False):
                if cid not in pos:
                    continue
                rows = g.index.tolist()
                slots = (g.clip_idx.astype(int).tolist() if mod == "voice" and "clip_idx" in g else list(range(len(rows))))
                for r, s in zip(rows, slots):
                    if s < K:
                        arr[pos[cid], s] = X[r]
                        m[pos[cid], s] = True
            B[mod], B[f"{mod}_mask"] = arr, m
    # questionnaire (M-CHAT items)
    qc = quest_cols if quest_cols is not None else [c for c in ch.columns if c.startswith("mchat_") and c != "mchat_total"]
    if qc:
        q = ch[qc].astype(float)
        B["quest"] = np.nan_to_num(q.values).astype(np.float32)
        B["quest_mask"] = q.notna().all(1).values
    if out_path:
        np.savez_compressed(out_path, **{k: v for k, v in B.items() if isinstance(v, np.ndarray)},
                            face_dims=np.array(B.get("face_dims", (0, 0))))
    return B


def load_bundle(path: Path) -> Dict:
    z = np.load(path, allow_pickle=True)  # our own file
    B = {k: z[k] for k in z.files}
    B["child_id"] = B["child_id"].astype(str)
    B["face_dims"] = tuple(int(v) for v in B.get("face_dims", np.array([0, 0])))
    return B


def bundle_dims(B: Dict) -> Dict[str, int]:
    d = {}
    if "face" in B:
        d["face"] = B["face"].shape[-1]
    for m in ("voice", "cry", "quest"):
        if m in B:
            d[m] = B[m].shape[-1]
    return d


def fit_norm(B: Dict, train_idx: np.ndarray) -> Dict[str, tuple]:
    """Per-modality mean/sd from training children only."""
    norm = {}
    for m in ("face", "voice", "cry", "quest"):
        if m not in B:
            continue
        mask = B[f"{m}_frame_mask"] if m == "face" else B[f"{m}_mask"]
        sel = B[m][train_idx][mask[train_idx]].astype(np.float32)
        if len(sel):
            norm[m] = (sel.mean(0), sel.std(0) + 1e-5)
    return norm


def apply_norm(B: Dict, norm: Dict[str, tuple]) -> Dict:
    out = dict(B)
    for m, (mu, sd) in norm.items():
        out[m] = ((B[m].astype(np.float32) - mu) / sd).astype(np.float32)
        if m == "face":
            out[m] *= B["face_frame_mask"][..., None]
    return out


def standardise_bundle(B: Dict, train_idx: np.ndarray) -> Dict:
    """Z-score each modality with statistics from training children only."""
    return apply_norm(B, fit_norm(B, train_idx))


def batch_from_bundle(B: Dict, idx, dev, mods=MODALITIES) -> Dict[str, torch.Tensor]:
    b = {}
    for m in mods:
        if m not in B:
            continue
        b[m] = torch.as_tensor(B[m][idx], dtype=torch.float32, device=dev)
        b[f"{m}_mask"] = torch.as_tensor(B[f"{m}_mask"][idx], device=dev)
        if m == "face":
            b["face_frame_mask"] = torch.as_tensor(B["face_frame_mask"][idx], device=dev)
    return b


# ----------------------------------------------------------------------------
# proposed model: XMF
# ----------------------------------------------------------------------------
class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))

    def forward(self, x, mask):  # x [B,L,d], mask [B,L] bool
        s = self.q(x).squeeze(-1).float()
        s = s.masked_fill(~mask, -1e4)
        a = torch.softmax(s, -1) * mask.float()
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-6)
        return (a.unsqueeze(-1) * x).sum(1), a


class TemporalFace(nn.Module):
    """Per-clip temporal transformer over frames (2 layers)."""

    def __init__(self, din, d, layers=2, heads=4, dropout=0.1, max_t=64, temporal=True):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(din, d), nn.GELU(), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.zeros(1, max_t, d))
        self.temporal = temporal
        if temporal:
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers)
        self.pool = AttnPool(d)

    def forward(self, x, mask):  # x [B*C,T,din], mask [B*C,T]
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        safe = mask.clone()
        safe[safe.sum(-1) == 0, 0] = True  # avoid all-masked rows
        if self.temporal:
            h = self.enc(h, src_key_padding_mask=~safe)
        return self.pool(h, safe)  # clip token, frame attention


class CrossBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, q, kv, kv_mask):
        safe = kv_mask.clone()
        safe[safe.sum(-1) == 0, 0] = True
        a, w = self.att(self.ln1(q), kv, kv, key_padding_mask=~safe, need_weights=True)
        has = (kv_mask.sum(-1) > 0).float()[:, None, None]
        q = q + a * has
        return q + self.ff(self.ln2(q)), w


class XMF(nn.Module):
    """BanglaASD-XMF: four light encoders -> learned missing tokens ->
    bidirectional cross-modal attention (face <-> audio) -> clip->child
    attention pooling per modality -> gated fusion -> evidential head that
    returns P(ASD) and an uncertainty used for 'refer'."""

    def __init__(self, dims: Dict[str, int], d: int = 256, heads: int = 4, temporal_layers: int = 2,
                 dropout: float = 0.2, max_clips: int = 10, temporal: bool = True, cross: bool = True,
                 evidential: bool = True, use_mods: Sequence[str] = MODALITIES, face_slice: Optional[slice] = None):
        super().__init__()
        self.mods = [m for m in MODALITIES if m in dims and m in use_mods]
        self.evidential, self.cross, self.face_slice = evidential, cross, face_slice
        self.d = d
        if "face" in self.mods:
            din = dims["face"] if face_slice is None else len(range(*face_slice.indices(dims["face"])))
            self.face_enc = TemporalFace(din, d, temporal_layers, heads, dropout / 2, temporal=temporal)
        self.enc = nn.ModuleDict({m: nn.Sequential(nn.Linear(dims[m], d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(dropout))
                                  for m in self.mods if m != "face"})
        self.missing = nn.ParameterDict({m: nn.Parameter(torch.randn(d) * 0.02) for m in self.mods})
        self.type_emb = nn.ParameterDict({m: nn.Parameter(torch.randn(d) * 0.02) for m in self.mods})
        self.clip_pos = nn.Parameter(torch.zeros(1, max_clips, d))
        if cross:
            self.f2a = CrossBlock(d, heads, dropout)
            self.a2f = CrossBlock(d, heads, dropout)
        self.pools = nn.ModuleDict({m: AttnPool(d) for m in self.mods})
        self.gate = nn.Sequential(nn.Linear(d * len(self.mods), d), nn.GELU(), nn.Linear(d, len(self.mods)))
        self.mix = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(d, 2 if evidential else 1)

    def _tokens(self, b, force_missing=()):
        toks, masks, extra = {}, {}, {}
        for m in self.mods:
            if m not in b:
                continue
            if m == "face":
                x, fm = b["face"], b["face_frame_mask"]
                if self.face_slice is not None:
                    x = x[..., self.face_slice]
                Bn, C, T, D = x.shape
                tok, fa = self.face_enc(x.reshape(Bn * C, T, D), fm.reshape(Bn * C, T))
                toks[m] = tok.view(Bn, C, -1) + self.clip_pos[:, :C]
                extra["frame_attn"] = fa.view(Bn, C, T)
                mk = b["face_mask"].bool()
            elif m == "quest":
                toks[m] = self.enc[m](b["quest"]).unsqueeze(1)
                mk = b["quest_mask"].bool().unsqueeze(1)
            else:
                t = self.enc[m](b[m])
                if m == "voice":
                    t = t + self.clip_pos[:, : t.shape[1]]
                toks[m] = t
                mk = b[f"{m}_mask"].bool()
            if m in force_missing:
                mk = torch.zeros_like(mk)
            toks[m] = toks[m] + self.type_emb[m]
            masks[m] = mk
        return toks, masks, extra

    def forward(self, b, force_missing=(), drop_mods: Optional[torch.Tensor] = None, return_all=False):
        toks, masks, extra = self._tokens(b, force_missing)
        present = [m for m in self.mods if m in toks]
        if drop_mods is not None:  # [B, n_mods] bool -> drop that modality for that child (training)
            for j, m in enumerate(self.mods):
                if m in masks:
                    masks[m] = masks[m] & ~drop_mods[:, j:j + 1]
        if self.cross and "face" in toks:
            audio = [m for m in ("voice", "cry") if m in toks]
            if audio:
                A = torch.cat([toks[m] for m in audio], 1)
                Am = torch.cat([masks[m] for m in audio], 1)
                newF, wf = self.f2a(toks["face"], A, Am)
                newA, wa = self.a2f(A, toks["face"], masks["face"])
                toks["face"] = newF
                off = 0
                for m in audio:
                    L = toks[m].shape[1]
                    toks[m] = newA[:, off:off + L]
                    off += L
                extra["cross_attn_face"] = wf
        pooled, avail, clip_attn = [], [], {}
        for m in present:
            h, a = self.pools[m](toks[m], masks[m])
            has = masks[m].any(-1)
            h = torch.where(has.unsqueeze(-1), h, self.missing[m].expand_as(h))
            pooled.append(h)
            avail.append(has)
            clip_attn[m] = a
        H = torch.stack(pooled, 1)  # B,M,d
        av = torch.stack(avail, 1)  # B,M
        g = self.gate(H.flatten(1)).float().masked_fill(~av, -1e4)
        g = torch.softmax(g, -1)
        g = torch.where(av.any(-1, keepdim=True), g, torch.full_like(g, 1.0 / g.shape[1]))
        z = (g.unsqueeze(-1) * H).sum(1)
        z = z + self.mix(z)
        out = self.head(z).float()
        if self.evidential:
            alpha = F.softplus(out) + 1
            S = alpha.sum(-1)
            prob = alpha[:, 1] / S
            unc = 2.0 / S
        else:
            alpha = None
            prob = torch.sigmoid(out.squeeze(-1))
            unc = torch.zeros_like(prob)
        res = {"prob": prob, "unc": unc, "alpha": alpha, "logit": out, "gates": g, "z": z, "avail": av}
        if return_all:
            res.update(extra)
            res["clip_attn"] = clip_attn
            res["mods"] = present
        return res


def evidential_loss(alpha, y, epoch, anneal=10, w_pos: float = 1.0):
    """Sensoy et al. 2018 (MSE form) + annealed KL to the uniform Dirichlet."""
    yoh = F.one_hot(y.long(), 2).float()
    S = alpha.sum(-1, keepdim=True)
    p = alpha / S
    err = ((yoh - p) ** 2).sum(-1)
    var = (p * (1 - p) / (S + 1)).sum(-1)
    at = yoh + (1 - yoh) * alpha
    St = at.sum(-1)
    kl = (torch.lgamma(St) - math.lgamma(2) - torch.lgamma(at).sum(-1)
          + ((at - 1) * (torch.digamma(at) - torch.digamma(St.unsqueeze(-1)))).sum(-1))
    w = torch.where(y > 0.5, torch.full_like(err, w_pos), torch.ones_like(err))
    return (w * (err + var + min(1.0, epoch / anneal) * kl)).mean()


# ----------------------------------------------------------------------------
# fusion baselines
# ----------------------------------------------------------------------------
def pooled_inputs(b: Dict[str, torch.Tensor], mods) -> Dict[str, torch.Tensor]:
    """Masked mean over frames and clips -> one vector per modality per child."""
    out = {}
    for m in mods:
        if m not in b:
            continue
        if m == "face":
            fm = b["face_frame_mask"].float().unsqueeze(-1)
            clip = (b["face"] * fm).sum(2) / fm.sum(2).clamp_min(1)
            cm = b["face_mask"].float().unsqueeze(-1)
            out[m] = (clip * cm).sum(1) / cm.sum(1).clamp_min(1)
        elif m == "quest":
            out[m] = b["quest"] * b["quest_mask"].float().unsqueeze(-1)
        else:
            mk = b[f"{m}_mask"].float().unsqueeze(-1)
            out[m] = (b[m] * mk).sum(1) / mk.sum(1).clamp_min(1)
    return out


def avail_of(b, m):
    if m == "quest":
        return b["quest_mask"].bool()
    return b[f"{m}_mask"].bool().any(-1)


class EarlyConcat(nn.Module):
    def __init__(self, dims, d=256, dropout=0.3, mods=MODALITIES):
        super().__init__()
        self.mods = [m for m in MODALITIES if m in dims and m in mods]
        din = sum(dims[m] for m in self.mods) + len(self.mods)
        self.net = nn.Sequential(nn.Linear(din, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, b, force_missing=(), drop_mods=None, return_all=False):
        P = pooled_inputs(b, self.mods)
        xs = []
        for j, m in enumerate(self.mods):
            a = avail_of(b, m)
            if m in force_missing:
                a = torch.zeros_like(a)
            if drop_mods is not None:
                a = a & ~drop_mods[:, j]
            xs += [P[m] * a.float().unsqueeze(-1), a.float().unsqueeze(-1)]
        logit = self.net(torch.cat(xs, -1)).float().squeeze(-1)
        p = torch.sigmoid(logit)
        return {"prob": p, "unc": torch.zeros_like(p), "logit": logit}


class LateAverage(nn.Module):
    def __init__(self, dims, d=128, dropout=0.3, mods=MODALITIES):
        super().__init__()
        self.mods = [m for m in MODALITIES if m in dims and m in mods]
        self.heads = nn.ModuleDict({m: nn.Sequential(nn.Linear(dims[m], d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
                                    for m in self.mods})

    def forward(self, b, force_missing=(), drop_mods=None, return_all=False):
        P = pooled_inputs(b, self.mods)
        logits, avs = [], []
        for j, m in enumerate(self.mods):
            a = avail_of(b, m)
            if m in force_missing:
                a = torch.zeros_like(a)
            if drop_mods is not None:
                a = a & ~drop_mods[:, j]
            logits.append(self.heads[m](P[m]).float().squeeze(-1))
            avs.append(a)
        L = torch.stack(logits, 1)
        A = torch.stack(avs, 1).float()
        probs = torch.sigmoid(L)
        p = (probs * A).sum(1) / A.sum(1).clamp_min(1)
        p = torch.where(A.sum(1) > 0, p, torch.full_like(p, 0.5))
        return {"prob": p, "unc": (probs.var(1) if probs.shape[1] > 1 else torch.zeros_like(p)), "per_mod_logits": L,
                "per_mod_avail": A}


class _Projector(nn.Module):
    def __init__(self, dims, k, mods):
        super().__init__()
        self.mods = mods
        self.proj = nn.ModuleDict({m: nn.Sequential(nn.Linear(dims[m], k), nn.GELU()) for m in mods})
        self.missing = nn.ParameterDict({m: nn.Parameter(torch.zeros(k)) for m in mods})

    def forward(self, b, force_missing, drop_mods):
        P = pooled_inputs(b, self.mods)
        out = []
        for j, m in enumerate(self.mods):
            a = avail_of(b, m)
            if m in force_missing:
                a = torch.zeros_like(a)
            if drop_mods is not None:
                a = a & ~drop_mods[:, j]
            h = self.proj[m](P[m])
            out.append(torch.where(a.unsqueeze(-1), h, self.missing[m].expand_as(h)))
        return out


class TFN(nn.Module):
    """Tensor Fusion Network (Zadeh 2017): outer product of [h;1]."""

    def __init__(self, dims, k=None, dropout=0.3, mods=MODALITIES):
        super().__init__()
        self.mods = [m for m in MODALITIES if m in dims and m in mods]
        k = k or {1: 64, 2: 48, 3: 24, 4: 12}[len(self.mods)]
        self.p = _Projector(dims, k, self.mods)
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear((k + 1) ** len(self.mods), 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, b, force_missing=(), drop_mods=None, return_all=False):
        hs = self.p(b, force_missing, drop_mods)
        Bn = hs[0].shape[0]
        z = torch.ones(Bn, 1, device=hs[0].device)
        for h in hs:
            h1 = torch.cat([h, torch.ones(Bn, 1, device=h.device)], -1)
            z = torch.einsum("bi,bj->bij", z, h1).reshape(Bn, -1)
        logit = self.net(z).float().squeeze(-1)
        p = torch.sigmoid(logit)
        return {"prob": p, "unc": torch.zeros_like(p), "logit": logit}


class LMF(nn.Module):
    """Low-rank Multimodal Fusion (Liu 2018)."""

    def __init__(self, dims, k=64, rank=4, out=64, mods=MODALITIES, dropout=0.3):
        super().__init__()
        self.mods = [m for m in MODALITIES if m in dims and m in mods]
        self.p = _Projector(dims, k, self.mods)
        self.factors = nn.ParameterList([nn.Parameter(torch.randn(rank, k + 1, out) * 0.05) for _ in self.mods])
        self.w = nn.Parameter(torch.randn(1, rank) * 0.1)
        self.head = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(out, 1))

    def forward(self, b, force_missing=(), drop_mods=None, return_all=False):
        hs = self.p(b, force_missing, drop_mods)
        Bn = hs[0].shape[0]
        prod = None
        for h, Wf in zip(hs, self.factors):
            h1 = torch.cat([h, torch.ones(Bn, 1, device=h.device)], -1)
            f = torch.einsum("bk,rko->bro", h1, Wf)
            prod = f if prod is None else prod * f
        z = torch.einsum("xr,bro->bo", self.w, prod)
        logit = self.head(z).float().squeeze(-1)
        p = torch.sigmoid(logit)
        return {"prob": p, "unc": torch.zeros_like(p), "logit": logit}


class MulTLite(nn.Module):
    """MulT-style (Tsai 2019) crossmodal transformer: every modality's token
    sequence attends to every other modality; then self-pool and concat."""

    def __init__(self, dims, d=128, heads=4, dropout=0.2, mods=MODALITIES):
        super().__init__()
        self.inner = XMF(dims, d=d, heads=heads, temporal_layers=1, dropout=dropout, cross=False, evidential=False, use_mods=mods)
        self.mods = self.inner.mods
        self.blocks = nn.ModuleDict({f"{a}_{b}": CrossBlock(d, heads, dropout) for a in self.mods for b in self.mods if a != b})
        self.head = nn.Sequential(nn.Linear(d * len(self.mods), d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, b, force_missing=(), drop_mods=None, return_all=False):
        toks, masks, _ = self.inner._tokens(b, force_missing)
        if drop_mods is not None:
            for j, m in enumerate(self.mods):
                if m in masks:
                    masks[m] = masks[m] & ~drop_mods[:, j:j + 1]
        outs = []
        for a in self.mods:
            q = toks[a]
            for c in self.mods:
                if c != a:
                    q, _ = self.blocks[f"{a}_{c}"](q, toks[c], masks[c])
            h, _ = self.inner.pools[a](q, masks[a])
            has = masks[a].any(-1, keepdim=True)
            outs.append(torch.where(has, h, self.inner.missing[a].expand_as(h)))
        logit = self.head(torch.cat(outs, -1)).float().squeeze(-1)
        p = torch.sigmoid(logit)
        return {"prob": p, "unc": torch.zeros_like(p), "logit": logit}


def create_fusion_model(name: str, dims: Dict[str, int], cfg: Dict) -> nn.Module:
    mods = cfg.get("mods", MODALITIES)
    if name == "xmf":
        fs = None
        if cfg.get("no_handcrafted") and cfg.get("face_dims"):
            fs = slice(0, cfg["face_dims"][0])
        return XMF(dims, d=cfg.get("d", 256), heads=cfg.get("heads", 4), temporal_layers=cfg.get("temporal_layers", 2),
                   dropout=cfg.get("dropout", 0.2), temporal=not cfg.get("no_temporal", False),
                   cross=not cfg.get("no_cross", False), evidential=not cfg.get("no_evidential", False),
                   use_mods=mods, face_slice=fs)
    if name == "early_concat":
        return EarlyConcat(dims, mods=mods)
    if name == "late_avg":
        return LateAverage(dims, mods=mods)
    if name == "tfn":
        return TFN(dims, mods=mods)
    if name == "lmf":
        return LMF(dims, mods=mods)
    if name == "mult":
        return MulTLite(dims, mods=mods)
    raise ValueError(name)


def mc_dropout_predict(model, b, n=20):
    """Monte-Carlo dropout: mean prob and std over n stochastic passes."""
    model.train()
    with torch.no_grad():
        ps = torch.stack([model(b)["prob"] for _ in range(n)], 0)
    model.eval()
    return ps.mean(0), ps.std(0)


def modality_shapley(model, b, mods: Sequence[str]) -> np.ndarray:
    """Exact Shapley value of each modality for P(ASD), per child, by
    masking modalities with their learned missing tokens (2^M subsets).
    Returns [B, M]; unavailable modalities get 0."""
    model.eval()
    M = len(mods)
    vals = {}
    with torch.no_grad():
        for r in range(M + 1):
            for S in itertools.combinations(range(M), r):
                miss = tuple(mods[j] for j in range(M) if j not in S)
                vals[S] = model(b, force_missing=miss)["prob"].float().cpu().numpy()
    Bn = next(iter(vals.values())).shape[0]
    phi = np.zeros((Bn, M))
    for j in range(M):
        others = [k for k in range(M) if k != j]
        for r in range(M):
            for S in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(M - r - 1) / math.factorial(M)
                with_j = tuple(sorted(S + (j,)))
                phi[:, j] += w * (vals[with_j] - vals[tuple(S)])
    return phi


# ----------------------------------------------------------------------------
# fusion training task (NB08 / NB09)
# ----------------------------------------------------------------------------
def _stack_sklearn(B, tr, va, te, mods, seed):
    """Stacking baseline: per-modality logistic regression (OOF inside the
    training children) -> meta logistic regression on modality probs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    dev = torch.device("cpu")
    allidx = np.arange(len(B["y"]))
    P = pooled_inputs(batch_from_bundle(B, allidx, dev, mods), mods)
    feats = {m: P[m].numpy() for m in P}
    av = {m: avail_of(batch_from_bundle(B, allidx, dev, (m,)), m).numpy() for m in P}
    y = B["y"]
    meta_tr = np.full((len(tr), len(feats)), 0.5)
    meta_all = np.full((len(y), len(feats)), 0.5)
    for j, m in enumerate(feats):
        X = feats[m]
        tr_av = tr[av[m][tr]]
        if len(np.unique(y[tr_av])) < 2:
            continue
        k_in = int(min(5, np.bincount(y[tr_av]).min()))
        if k_in < 2:
            continue
        skf = StratifiedKFold(k_in, shuffle=True, random_state=seed)
        oof = np.full(len(tr), 0.5)
        pos_in_tr = {c: i for i, c in enumerate(tr)}
        for a, bb in skf.split(tr_av, y[tr_av]):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            clf.fit(X[tr_av[a]], y[tr_av[a]])
            for k, pr in zip(tr_av[bb], clf.predict_proba(X[tr_av[bb]])[:, 1]):
                oof[pos_in_tr[k]] = pr
        meta_tr[:, j] = oof
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X[tr_av], y[tr_av])
        meta_all[:, j] = np.where(av[m], clf.predict_proba(X)[:, 1], 0.5)
    meta = LogisticRegression(C=1.0, max_iter=2000).fit(np.c_[meta_tr, [[av[m][i] for m in feats] for i in tr]], y[tr])
    Xall = np.c_[meta_all, np.stack([av[m] for m in feats], 1)]
    return meta.predict_proba(Xall)[:, 1]


def explode_to_clips(B: Dict) -> Dict:
    """Child bundle -> one sample per available face clip (clip-level
    ablation).  Child-level modalities (cry, questionnaire) are repeated."""
    keep = np.argwhere(B["face_mask"])  # (child, clip)
    i, c = keep[:, 0], keep[:, 1]
    E = {"child_id": B["child_id"][i], "y": B["y"][i], "face_dims": B.get("face_dims", (0, 0))}
    E["face"] = B["face"][i, c][:, None]
    E["face_frame_mask"] = B["face_frame_mask"][i, c][:, None]
    E["face_mask"] = np.ones((len(i), 1), bool)
    if "voice" in B:
        E["voice"] = B["voice"][i, c][:, None]
        E["voice_mask"] = B["voice_mask"][i, c][:, None]
    for m in ("cry", "quest"):
        if m in B:
            E[m] = B[m][i]
            E[f"{m}_mask"] = B[f"{m}_mask"][i]
    return E


def fusion_fold_task(job: Dict) -> None:
    out = Path(job["out_dir"])
    flag = out / "flags" / f"{job['name']}.done"
    if flag.exists():
        return
    U.seed_everything(job["seed"])
    dev = device()
    B = load_bundle(Path(job["bundle"]))
    cfg = job.get("cfg", {})
    if cfg.get("clip_level"):
        B = explode_to_clips(B)
    children = pd.read_csv(job["manifest"])
    tr_ids, va_ids, te_ids = U.fold_ids(children, job["seed"], job["fold"])
    U.assert_no_leak(tr_ids, va_ids, te_ids)
    cid = B["child_id"]
    tr = np.flatnonzero(np.isin(cid, tr_ids))
    va = np.flatnonzero(np.isin(cid, va_ids))
    te = np.flatnonzero(np.isin(cid, te_ids))
    mods = tuple(m for m in cfg.get("mods", MODALITIES) if m in B)
    y = B["y"]

    def write(probs_va, probs_te, unc_va=None, unc_te=None, extra=None):
        extra = extra or {}
        for part, idx, pr, un in (("val", va, probs_va, unc_va), ("test", te, probs_te, unc_te)):
            if len(idx) == 0:
                continue
            d = pd.DataFrame({"child_id": B["child_id"][idx], "y_true": y[idx], "prob": pr,
                              "uncertainty": un if un is not None else np.nan, "part": part})
            if cfg.get("clip_level"):  # clip-level AUROC, then aggregate clips -> child
                extra[f"clip_auc_{part}"] = U._safe_auc(d.y_true, d.prob)
                d = d.groupby(["child_id", "part"], as_index=False).agg(y_true=("y_true", "first"), prob=("prob", "mean"),
                                                                       uncertainty=("prob", "std"))
            U.save_preds(d, job["model_tag"], job["fold"], out / "preds", job["seed"])
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text(json.dumps(extra or {}))

    if job["model"] == "stacking":
        p = _stack_sklearn(B, tr, va, te, mods, job["seed"])
        write(p[va], p[te])
        return

    # inner early-stopping split (keeps the OOF fold clean)
    rng = np.random.default_rng(job["seed"] * 100 + job["fold"])
    tr_child = pd.DataFrame({"c": cid[tr], "y": y[tr]}).drop_duplicates("c")
    inner_c = []
    for cls in (0, 1):
        cc = tr_child.c[tr_child.y == cls].to_numpy()
        if len(cc):
            inner_c += list(rng.choice(cc, size=max(1, int(round(0.15 * len(cc)))), replace=False))
    inner = tr[np.isin(cid[tr], inner_c)]  # chosen by child, so clip-level mode cannot leak
    fit = np.setdiff1d(tr, inner)
    norm = fit_norm(B, fit)
    Bs = apply_norm(B, norm)
    dims = bundle_dims(Bs)
    if cfg.get("no_handcrafted"):
        cfg["face_dims"] = tuple(B["face_dims"])
    model = create_fusion_model(job["model"], {m: dims[m] for m in mods}, cfg).to(dev)
    mods_model = list(getattr(model, "mods", mods))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("wd", 0.05))
    epochs, bs = cfg.get("epochs", 120), cfg.get("batch", 32)
    steps_per = max(1, int(math.ceil(len(fit) / bs)))
    total = epochs * steps_per
    p_md, max_drop = cfg.get("modality_dropout", 0.5), cfg.get("max_drop", 2)
    evid = isinstance(model, XMF) and model.evidential
    w_pos = float((y[fit] == 0).sum() / max(1, (y[fit] == 1).sum()))
    best, best_ep, patience, step = -1, -1, cfg.get("patience", 25), 0
    best_state = None

    def predict(idx, force_missing=()):
        model.eval()
        ps, us = [], []
        with torch.no_grad():
            for s in range(0, len(idx), 128):
                o = model(batch_from_bundle(Bs, idx[s:s + 128], dev, mods_model), force_missing=force_missing)
                ps.append(o["prob"].float().cpu().numpy())
                us.append(o["unc"].float().cpu().numpy())
        return np.concatenate(ps) if ps else np.array([]), np.concatenate(us) if us else np.array([])

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(fit)
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            b = batch_from_bundle(Bs, idx, dev, mods_model)
            drop = None
            if p_md > 0 and len(mods_model) > 1:
                av = torch.stack([avail_of(b, m) for m in mods_model], 1)
                drop = torch.zeros_like(av)
                for i in range(len(idx)):
                    if rng.random() < p_md:
                        have = torch.nonzero(av[i]).flatten().tolist()
                        if len(have) > 1:
                            k = int(rng.integers(1, min(max_drop, len(have) - 1) + 1))
                            for j in rng.choice(have, size=k, replace=False):
                                drop[i, j] = True
            cosine_lr(opt, [cfg.get("lr", 3e-4)], step, total, warmup=steps_per * 3)
            o = model(b, drop_mods=drop)
            yt = torch.as_tensor(y[idx], device=dev)
            if evid:
                loss = evidential_loss(o["alpha"], yt, ep, anneal=cfg.get("anneal", 20), w_pos=min(3, w_pos))
            elif "per_mod_logits" in o:  # late average: train every head on its own modality
                L, A = o["per_mod_logits"], o["per_mod_avail"]
                bce = F.binary_cross_entropy_with_logits(L, yt.float().unsqueeze(1).expand_as(L), reduction="none")
                loss = (bce * A).sum() / A.sum().clamp_min(1)
            else:
                loss = F.binary_cross_entropy_with_logits(o["logit"].view(-1), yt.float(),
                                                          pos_weight=torch.tensor(min(3, w_pos), device=dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        if ep % cfg.get("eval_every", 2) == 0 or ep == epochs - 1:
            pi, _ = predict(inner)
            auc = U._safe_auc(y[inner], pi)
            auc = -1 if np.isnan(auc) else auc
            if auc > best + 1e-4:
                best, best_ep = auc, ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif ep - best_ep >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    # temperature scaling on the inner split, applied to val/test
    pi, _ = predict(inner)
    ts = U.TemperatureScaler().fit(y[inner], pi)
    pv, uv = predict(va)
    pt, ut = predict(te)
    pv_cal, pt_cal = ts.transform(pv), ts.transform(pt) if len(pt) else pt
    extra = {"best_inner_auc": best, "best_ep": best_ep, "T": ts.T}
    # test-time missing-modality robustness (NB09)
    if job.get("missing_eval") and len(te):
        rob = {}
        for r in range(0, len(mods_model)):
            for S in itertools.combinations(mods_model, r):
                p_, _ = predict(te, force_missing=S)
                rob["-".join(S) or "none"] = U._safe_auc(y[te], p_)
        extra["missing_auc"] = rob
    if job.get("save_model"):
        (out / "ckpt").mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "cfg": cfg, "dims": {m: dims[m] for m in mods_model},
                    "mods": mods_model, "T": ts.T, "name": job["model"], "seed": job["seed"], "fold": job["fold"],
                    "norm": {m: (mu.tolist(), sd.tolist()) for m, (mu, sd) in norm.items()},
                    "fit_children": B["child_id"][fit].tolist()}, out / "ckpt" / f"{job['name']}.pt")
    write(pv_cal, pt_cal, uv, ut, extra)


# ----------------------------------------------------------------------------
# NB03: facial-expression pre-training of the frame backbones
# ----------------------------------------------------------------------------
EXPR_CLASSES = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"]


class ExprNet(nn.Module):
    """timm backbone -> 256-d expression embedding -> 7 classes.  The 256-d
    embedding is what NB05 caches for every cohort frame."""

    def __init__(self, base: str, n_cls: int = 7, emb: int = 256, pretrained: bool = True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(TIMM_ZOO[base], pretrained=pretrained, num_classes=0)
        with torch.no_grad():  # num_features is wrong for some timm models (e.g. MobileNetV3 head conv)
            was = self.backbone.training
            nf = self.backbone.eval()(torch.zeros(1, 3, 64, 64)).shape[-1]
            self.backbone.train(was)
        self.emb_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(nf, emb), nn.BatchNorm1d(emb), nn.GELU())
        self.cls_head = nn.Linear(emb, n_cls)

    def embed(self, x):
        return self.emb_head(self.backbone(x))

    def forward(self, x):
        return self.cls_head(self.embed(x))


def load_expr_net(base: str, ckpt_dir: Path) -> Optional[ExprNet]:
    ckpt_dir = Path(ckpt_dir)
    bb, eh = ckpt_dir / f"{base}_expr_backbone.pt", ckpt_dir / f"{base}_expr_head.pt"
    if not bb.exists():
        return None
    m = ExprNet(base, pretrained=False)
    m.backbone.load_state_dict(torch.load(bb, map_location="cpu"))
    if eh.exists():
        sd = torch.load(eh, map_location="cpu")
        m.emb_head.load_state_dict(sd["emb_head"])
        m.cls_head.load_state_dict(sd["cls_head"])
    return m.eval()


class MemmapImages(torch.utils.data.Dataset):
    """uint8 image memmap opened lazily in each worker (picklable)."""

    def __init__(self, path, labels, ids):
        self.path, self.labels, self.ids, self.X = str(path), labels, ids, None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.X is None:
            self.X = np.load(self.path, mmap_mode="r")
        j = self.ids[i]
        return torch.from_numpy(np.ascontiguousarray(self.X[j])), int(self.labels[j])


def expr_pretrain_task(job: Dict) -> None:
    """Train one backbone on the cached FER2013 + RAF-DB (+ optional extra)
    arrays.  Early stopping on a held-out 5 % of training images; final
    report on each dataset's official test split."""
    from sklearn.metrics import accuracy_score, f1_score
    out = Path(job["out_dir"])
    flag = out / "flags" / f"{job['name']}.done"
    if flag.exists():
        return
    U.seed_everything(job["seed"])
    dev = device()
    cache = Path(job["cache"])
    X = np.load(cache / "images.npy", mmap_mode="r")
    y = np.load(cache / "labels.npy")
    split = np.load(cache / "split.npy", allow_pickle=True)
    src = np.load(cache / "source.npy", allow_pickle=True)
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    te = np.flatnonzero(split == "test")
    if job.get("limit"):
        rng = np.random.default_rng(0)
        tr = rng.choice(tr, min(len(tr), job["limit"]), replace=False)
        va, te = va[: job["limit"] // 4], te[: job["limit"] // 2]
    size = job["img_size"]
    model = ExprNet(job["base"]).to(dev).to(memory_format=torch.channels_last)
    params = [{"params": model.backbone.parameters(), "lr": job["lr"] * 0.5},
              {"params": list(model.emb_head.parameters()) + list(model.cls_head.parameters()), "lr": job["lr"]}]
    opt = torch.optim.AdamW(params, weight_decay=0.05)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = make_scaler()
    counts = np.bincount(y[tr], minlength=7).astype(float)
    w = torch.tensor(np.clip((counts.sum() / np.maximum(counts, 1) / 7) ** 0.5, 0.3, 3), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    bs = job["batch"]
    steps_per = len(tr) // bs
    total = job["epochs"] * steps_per
    step, best, best_ep = 0, -2.0, -1
    ck = out / "ckpt"
    ck.mkdir(parents=True, exist_ok=True)

    def loader(ids, shuffle):
        return torch.utils.data.DataLoader(MemmapImages(cache / "images.npy", y, ids), batch_size=bs if shuffle else bs * 2, shuffle=shuffle,
                                           num_workers=job.get("workers", 2), pin_memory=True, drop_last=shuffle,
                                           persistent_workers=False)

    def predict(ids):
        model.eval()
        P = []
        with torch.no_grad(), amp_ctx():
            for xb, _ in loader(ids, False):
                xb = gpu_prepare(xb.to(dev, non_blocking=True), size, train=False)
                P.append(model(xb).float().softmax(-1).cpu().numpy())
        return np.concatenate(P) if P else np.zeros((0, 7))

    log = []
    for ep in range(job["epochs"]):
        model.train()
        t0 = time.time()
        for xb, yb in loader(tr, True):
            xb = gpu_prepare(xb.to(dev, non_blocking=True), size, train=True, gray=True)
            yb = yb.to(dev)
            # mixup on half of the batches
            lam = 1.0
            if np.random.rand() < 0.5:
                lam = float(np.random.beta(0.4, 0.4))
                perm = torch.randperm(xb.shape[0], device=dev)
                xb = lam * xb + (1 - lam) * xb[perm]
            cosine_lr(opt, base_lrs, step, total, warmup=steps_per)
            with amp_ctx():
                logits = model(xb).float()
                loss = lam * crit(logits, yb) + ((1 - lam) * crit(logits, yb[perm]) if lam < 1 else 0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if job.get("max_steps") and step >= job["max_steps"]:
                break
        pv = predict(va)
        f1 = f1_score(y[va], pv.argmax(1), average="macro") if len(va) else 0
        log.append({"epoch": ep, "val_macro_f1": f1, "sec": time.time() - t0})
        print(f"[{job['name']}] ep{ep} val_macroF1={f1:.4f} {time.time()-t0:.0f}s", flush=True)
        if f1 > best:
            best, best_ep = f1, ep
            torch.save(model.backbone.state_dict(), ck / f"{job['base']}_expr_backbone.pt")
            torch.save({"emb_head": model.emb_head.state_dict(), "cls_head": model.cls_head.state_dict(),
                        "classes": EXPR_CLASSES, "img_size": size}, ck / f"{job['base']}_expr_head.pt")
        elif ep - best_ep >= job.get("patience", 5):
            break
        if job.get("max_steps") and step >= job["max_steps"]:
            break
    model = load_expr_net(job["base"], ck).to(dev).to(memory_format=torch.channels_last)
    pt = predict(te)
    rows = []
    for s in sorted(set(src[te])):
        m = src[te] == s
        rows.append({"backbone": job["base"], "test_set": s, "n": int(m.sum()),
                     "acc": accuracy_score(y[te][m], pt[m].argmax(1)),
                     "macro_f1": f1_score(y[te][m], pt[m].argmax(1), average="macro")})
    pd.DataFrame(rows).to_csv(out / f"expr_test_{job['base']}.csv", index=False)
    pd.DataFrame(log).to_csv(out / f"expr_log_{job['base']}.csv", index=False)
    np.save(out / f"expr_test_probs_{job['base']}.npy", pt.astype(np.float16))
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(json.dumps({"best_val_f1": best, "best_ep": best_ep, "test": rows}))


def load_fusion_ckpt(path, dev=None):
    """-> (model, ckpt dict with numpy norm)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = create_fusion_model(ck["name"], ck["dims"], dict(ck["cfg"]))
    model.load_state_dict(ck["state"])
    ck["norm"] = {m: (np.asarray(mu, np.float32), np.asarray(sd, np.float32)) for m, (mu, sd) in ck["norm"].items()}
    return model.to(dev or device()).eval(), ck
