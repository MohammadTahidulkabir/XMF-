"""
asd_audio.py - voice and cry pipeline (NB02, NB02P, NB07, NB07P).

  load_audio      : wav/mp3/mp4 -> mono 16 kHz float32 (ffmpeg for video)
  logmel          : 64-band log-mel, 10 ms hop  (cached as float16 .npy)
  egemaps         : openSMILE eGeMAPSv02 functionals (88-d); librosa
                    prosody fallback if openSMILE cannot be installed
  Wav2Vec2Embedder: facebook/wav2vec2-base, time-mean of every layer
  ASTEmbedder     : MIT AST (AudioSet) pooled embedding (768-d)
  CNN14Embedder   : PANNs CNN14 2048-d embedding (optional)
  melcnn_fold_task: EfficientNet-B0 on log-mel, one fold (GPU)

Raw audio is decoded once; all training reads cached arrays.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SR = 16000


def load_audio(path, sr: int = SR, max_s: Optional[float] = 30.0) -> np.ndarray:
    path = str(path)
    if path.lower().endswith((".mp4", ".mov", ".m4a", ".webm", ".3gp", ".mkv", ".aac", ".mp3")):
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
        if max_s:
            cmd[6:6] = ["-t", str(max_s)]
        out = subprocess.run(cmd, capture_output=True, timeout=120)
        wav = np.frombuffer(out.stdout, np.float32).copy()
    else:
        import soundfile as sf
        wav, s0 = sf.read(path, dtype="float32", always_2d=True)
        wav = wav.mean(1)
        if s0 != sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=s0, target_sr=sr)
        if max_s:
            wav = wav[: int(max_s * sr)]
    if len(wav) == 0:
        return np.zeros(sr // 2, np.float32)
    peak = np.abs(wav).max()
    return (wav / peak * 0.9 if peak > 0 else wav).astype(np.float32)


def voiced_fraction(wav: np.ndarray, sr: int = SR, thr_db: float = -35) -> float:
    hop = sr // 100
    frames = wav[: len(wav) // hop * hop].reshape(-1, hop)
    if len(frames) == 0:
        return 0.0
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-8)
    return float((db > thr_db).mean())


def logmel(wav: np.ndarray, sr: int = SR, n_mels: int = 64) -> np.ndarray:
    import librosa
    m = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=400, hop_length=160, n_mels=n_mels, fmin=50, fmax=8000)
    return librosa.power_to_db(m, ref=1.0, top_db=80).astype(np.float16)


def mfcc_stats(wav: np.ndarray, sr: int = SR, n: int = 40) -> np.ndarray:
    """Proposal features: 40 MFCCs + deltas -> mean and std over time (160-d)."""
    import librosa
    m = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n, n_fft=400, hop_length=160)
    d = librosa.feature.delta(m) if m.shape[1] >= 9 else np.zeros_like(m)
    return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)]).astype(np.float32)


class Egemaps:
    def __init__(self):
        self.kind = "none"
        try:
            import opensmile
            self.smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                         feature_level=opensmile.FeatureLevel.Functionals)
            self.kind = "egemaps_v02"
            self.names = list(self.smile.feature_names)
        except Exception as e:
            print("[audio] openSMILE unavailable -> librosa prosody fallback:", e)
            self.kind = "librosa_prosody"
            self.names = None

    def __call__(self, wav: np.ndarray, sr: int = SR) -> np.ndarray:
        if self.kind == "egemaps_v02":
            return self.smile.process_signal(wav, sr).values[0].astype(np.float32)
        import librosa
        f0, vflag, _ = librosa.pyin(wav, fmin=80, fmax=900, sr=sr, frame_length=1024)
        f0v = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        rms = librosa.feature.rms(y=wav)[0]
        zcr = librosa.feature.zero_crossing_rate(wav)[0]
        cent = librosa.feature.spectral_centroid(y=wav, sr=sr)[0]
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13)
        st = lambda a: [np.mean(a), np.std(a)] if len(a) else [0.0, 0.0]
        jitter = float(np.mean(np.abs(np.diff(f0v))) / (np.mean(f0v) + 1e-6)) if len(f0v) > 2 else 0.0
        v = (st(f0v) + [np.ptp(f0v) if len(f0v) else 0.0, float(np.mean(vflag)) if vflag is not None else 0.0, jitter]
             + st(rms) + st(zcr) + st(cent) + list(mfcc.mean(1)) + list(mfcc.std(1)))
        return np.asarray(v, np.float32)


class Wav2Vec2Embedder:
    """Time-mean of every hidden layer -> [n_layers+1, 768]."""

    def __init__(self, name: str = "facebook/wav2vec2-base", device: str = None):
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2Model
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.m = Wav2Vec2Model.from_pretrained(name).to(self.dev).eval()
        if self.dev == "cuda":
            self.m = self.m.half()

    def __call__(self, wavs: Sequence[np.ndarray]) -> np.ndarray:
        torch = self.torch
        out = []
        for w in wavs:  # variable length: one at a time (fast enough, avoids padding bias)
            w = w[: SR * 20]
            x = self.fe(w, sampling_rate=SR, return_tensors="pt").input_values.to(self.dev)
            if self.dev == "cuda":
                x = x.half()
            with torch.no_grad():
                hs = self.m(x, output_hidden_states=True).hidden_states
            out.append(torch.stack([h.float().mean(1)[0] for h in hs]).cpu().numpy().astype(np.float16))
        return np.stack(out)


class ASTEmbedder:
    def __init__(self, name: str = "MIT/ast-finetuned-audioset-10-10-0.4593", device: str = None):
        import torch
        from transformers import ASTFeatureExtractor, ASTModel
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = ASTFeatureExtractor.from_pretrained(name)
        self.m = ASTModel.from_pretrained(name).to(self.dev).eval()
        if self.dev == "cuda":
            self.m = self.m.half()

    def __call__(self, wavs: Sequence[np.ndarray], batch: int = 16) -> np.ndarray:
        torch = self.torch
        out = []
        for s in range(0, len(wavs), batch):
            x = self.fe([w[: SR * 10] for w in wavs[s:s + batch]], sampling_rate=SR, return_tensors="pt").input_values.to(self.dev)
            if self.dev == "cuda":
                x = x.half()
            with torch.no_grad():
                out.append(self.m(input_values=x).pooler_output.float().cpu().numpy())
        return np.concatenate(out).astype(np.float16)


class CNN14Embedder:
    def __init__(self, device: str = None):
        import torch
        from panns_inference import AudioTagging
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.at = AudioTagging(checkpoint_path=None, device=self.dev)

    def __call__(self, wavs: Sequence[np.ndarray]) -> np.ndarray:
        out = []
        for w in wavs:
            w32 = np.asarray(w[: SR * 10], np.float32)
            import librosa
            w32 = librosa.resample(w32, orig_sr=SR, target_sr=32000)
            _, emb = self.at.inference(w32[None, :])
            out.append(emb[0])
        return np.stack(out).astype(np.float16)


def extract_all(items: pd.DataFrame, path_col: str, root: Path, out_dir: Path, tag: str,
                use_w2v: bool = True, use_ast: bool = True, use_cnn14: bool = False,
                max_s: float = 20.0, limit: Optional[int] = None) -> pd.DataFrame:
    """For every row: log-mel .npy (per item), eGeMAPS, wav2vec2 layers,
    AST and (optionally) CNN14 embeddings -> <tag>_*.npy aligned with
    <tag>_index.csv.  Rows whose audio cannot be read are dropped."""
    out_dir = Path(out_dir)
    (out_dir / f"{tag}_mel").mkdir(parents=True, exist_ok=True)
    items = items.reset_index(drop=True)
    if limit:
        items = items.iloc[:limit]
    eg = Egemaps()
    wavs, keep, voiced, dur = [], [], [], []
    t0 = time.time()
    for i, r in items.iterrows():
        try:
            w = load_audio(Path(root) / r[path_col], max_s=max_s)
        except Exception as e:
            print("[audio] skip", r[path_col], e)
            continue
        wavs.append(w)
        keep.append(i)
        voiced.append(voiced_fraction(w))
        dur.append(len(w) / SR)
        np.save(out_dir / f"{tag}_mel" / f"{len(keep)-1:06d}.npy", logmel(w))
    print(f"[audio] {tag}: decoded {len(wavs)}/{len(items)} in {time.time()-t0:.0f}s")
    idx = items.loc[keep].reset_index(drop=True)
    idx["voiced_frac"], idx["dur_s"], idx["row"] = voiced, dur, np.arange(len(idx))
    feats = {"egemaps": np.stack([eg(w) for w in wavs]).astype(np.float32),
             "mfcc": np.stack([mfcc_stats(w) for w in wavs]).astype(np.float32)}
    print(f"[audio] {tag}: {eg.kind} done {feats['egemaps'].shape}")
    for flag, name, cls in ((use_w2v, "w2v", Wav2Vec2Embedder), (use_ast, "ast", ASTEmbedder), (use_cnn14, "cnn14", CNN14Embedder)):
        if not flag:
            continue
        try:
            t0 = time.time()
            emb = cls()
            feats[name] = emb(wavs)
            print(f"[audio] {tag}: {name} {feats[name].shape} {time.time()-t0:.0f}s")
            del emb
        except Exception as e:
            print(f"[audio] {tag}: {name} failed: {e}")
    for k, v in feats.items():
        np.save(out_dir / f"{tag}_{k}.npy", v)
    idx.to_csv(out_dir / f"{tag}_index.csv", index=False)
    (out_dir / f"{tag}_meta.json").write_text(json.dumps({"egemaps_kind": eg.kind, "features": {k: list(v.shape) for k, v in feats.items()}}))
    return idx


def fusion_audio_features(out_dir: Path, tag: str, w2v_layers=(4, 5, 6, 7, 8, 9)) -> Optional[np.ndarray]:
    """Per-item vector used by the fusion model: mean of mid wav2vec2 layers
    (768) + eGeMAPS (88) (+ AST 768 when present)."""
    out_dir = Path(out_dir)
    parts = []
    w = out_dir / f"{tag}_w2v.npy"
    if w.exists():
        W = np.load(w).astype(np.float32)
        parts.append(W[:, list(w2v_layers)].mean(1))
    e = out_dir / f"{tag}_egemaps.npy"
    if e.exists():
        parts.append(np.nan_to_num(np.load(e).astype(np.float32)))
    a = out_dir / f"{tag}_ast.npy"
    if a.exists():
        parts.append(np.load(a).astype(np.float32))
    return np.concatenate(parts, 1) if parts else None


# ----------------------------------------------------------------------------
# mel-CNN (GPU) - one fold
# ----------------------------------------------------------------------------
class _MelCNNLSTMBase:
    pass


def _make_mel_cnn_lstm():
    import torch.nn as nn

    class MelCNNLSTM(nn.Module):
        """Proposal's CNN-LSTM: conv blocks over (freq, time) -> BiLSTM over time -> attention pooling."""

        def __init__(self, n_cls, c=64, h=128):
            super().__init__()
            blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(), nn.MaxPool2d((2, 1)))
            self.cnn = nn.Sequential(blk(1, c // 2), blk(c // 2, c), blk(c, c), nn.Dropout(0.2))
            self.lstm = nn.LSTM(c * 8, h, batch_first=True, bidirectional=True)
            self.att = nn.Linear(2 * h, 1)
            self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * h, n_cls))

        def forward(self, x):  # x [B,1,64,T]
            f = self.cnn(x)                      # B,c,8,T
            f = f.flatten(1, 2).transpose(1, 2)  # B,T,c*8
            h, _ = self.lstm(f)
            a = self.att(h).softmax(1)
            return self.fc((a * h).sum(1))
    return MelCNNLSTM


def MelCNNLSTM(n_cls):
    return _make_mel_cnn_lstm()(n_cls)


def melcnn_fold_task(job: Dict) -> None:
    """EfficientNet-B0 (ImageNet init, 1 input channel) on random 3 s log-mel
    crops.  Multiclass or binary.  Item-level preds -> CSV."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm
    import asd_utils as U
    import asd_models as M
    out = Path(job["out_dir"])
    flag = out / "flags" / f"{job['name']}.done"
    if flag.exists():
        return
    U.seed_everything(job["seed"])
    dev = M.device()
    idx = pd.read_csv(job["index"])
    mel_dir = Path(job["mel_dir"])
    y_all = idx[job["label_col"]].values
    tr = np.array(job["train_rows"])
    te = np.array(job["test_rows"])
    classes = sorted(pd.unique(y_all[tr]).tolist())
    cmap = {c: i for i, c in enumerate(classes)}
    yi = np.array([cmap.get(c, 0) for c in y_all])
    n_cls = len(classes)
    W = job.get("crop_frames", 300)
    mels = {}

    def get(r):
        if r not in mels:
            mels[r] = np.load(mel_dir / f"{int(idx.row[r]):06d}.npy").astype(np.float32)
        return mels[r]

    def crop(m, train):
        if m.shape[1] < W:
            m = np.pad(m, ((0, 0), (0, W - m.shape[1])), constant_values=m.min())
        s = np.random.randint(0, m.shape[1] - W + 1) if train else (m.shape[1] - W) // 2
        return m[:, s:s + W]

    if job.get("arch", "effb0") == "cnn_lstm":
        net = MelCNNLSTM(n_cls).to(dev)
    else:
        net = timm.create_model("efficientnet_b0.ra_in1k", pretrained=True, in_chans=1, num_classes=n_cls).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=job.get("lr", 1e-3), weight_decay=0.02)
    scaler = M.make_scaler()
    counts = np.bincount(yi[tr], minlength=n_cls).astype(float)
    wts = torch.tensor((counts.sum() / np.maximum(counts, 1)) / n_cls, dtype=torch.float32, device=dev).clamp(0.2, 5)
    epochs, bs = job.get("epochs", 25), job.get("batch", 64)
    steps = epochs * max(1, len(tr) // bs)
    step = 0
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(tr)
        for s in range(0, len(perm) - bs + 1 if len(perm) >= bs else 1, bs):
            rows = perm[s:s + bs]
            x = torch.tensor(np.stack([crop(get(r), True) for r in rows]))[:, None].to(dev)
            x = (x + 40) / 20
            # SpecAugment: time + frequency masks
            for _ in range(2):
                f0 = np.random.randint(0, 56)
                x[:, :, f0:f0 + np.random.randint(0, 8)] = 0
                t0 = np.random.randint(0, W - 30)
                x[:, :, :, t0:t0 + np.random.randint(0, 30)] = 0
            yt = torch.tensor(yi[rows], device=dev)
            M.cosine_lr(opt, [job.get("lr", 1e-3)], step, steps, warmup=20)
            with M.amp_ctx():
                loss = F.cross_entropy(net(x).float(), yt, weight=wts, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if job.get("max_steps") and step >= job["max_steps"]:
                break
        if job.get("max_steps") and step >= job["max_steps"]:
            break
    net.eval()
    tr_eval = tr[np.random.default_rng(0).permutation(len(tr))[:2000]]
    with torch.no_grad(), M.amp_ctx():
        tp = [net((torch.tensor(np.stack([crop(get(r), False) for r in tr_eval[s:s + 128]]))[:, None].to(dev) + 40) / 20).float().argmax(1).cpu().numpy()
              for s in range(0, len(tr_eval), 128)]
    train_acc = float((np.concatenate(tp) == yi[tr_eval]).mean()) if len(tr_eval) else float("nan")
    probs = []
    with torch.no_grad(), M.amp_ctx():
        for s in range(0, len(te), 128):
            x = torch.tensor(np.stack([crop(get(r), False) for r in te[s:s + 128]]))[:, None].to(dev)
            probs.append(torch.softmax(net((x + 40) / 20).float(), -1).cpu().numpy())
    P = np.concatenate(probs) if probs else np.zeros((0, n_cls))
    df = pd.DataFrame(P, columns=[f"p_{c}" for c in classes])
    df.insert(0, "row", te)
    df["y_true"] = y_all[te]
    df["fold"] = job["fold"]
    df["train_acc"] = train_acc
    (out / "preds").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "preds" / f"{job['name']}.csv", index=False)
    if job.get("save_model"):
        (out / "ckpt").mkdir(parents=True, exist_ok=True)
        torch.save({"state": net.state_dict(), "classes": classes}, out / "ckpt" / f"{job['name']}.pt")
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("ok")
