"""
asd_utils.py - shared core for the BanglaASD-XMF notebook suite.

Every notebook imports this file (from the private Kaggle dataset
`asd-xmf-lib`) so that folds, metrics, statistics and file formats are
identical across all models.  Nothing here touches the GPU at import time.

Sections
  1. environment / paths / input discovery
  2. seeds, progress.json + resume
  3. child-wise splits (locked test set + stratified group K-fold x seeds)
  4. metrics (threshold-free + thresholded), calibration, decision curves
  5. statistics: bootstrap CIs by child, DeLong, McNemar, Holm
  6. fairness tables
  7. perceptual hash duplicate audit
  8. prediction files  preds_<model>_<fold>.csv
  9. parallel fold runner (one fold per GPU)
 10. synthetic DEV cohort (pipeline testing only - never a result)
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 1. environment / paths
# ----------------------------------------------------------------------------
IS_KAGGLE = os.path.isdir("/kaggle/input")
INPUT_ROOT = Path("/kaggle/input") if IS_KAGGLE else Path(os.environ.get("ASD_INPUT", "./_input"))
WORK = Path("/kaggle/working") if IS_KAGGLE else Path(os.environ.get("ASD_WORK", "./_work"))
WORK.mkdir(parents=True, exist_ok=True)
LIB_DIR = Path(__file__).resolve().parent

SYNTHETIC_BANNER = (
    "SYNTHETIC DEV COHORT - pipeline test only. These numbers are NOT results "
    "and must never appear in a paper."
)


def find_path(name: str, roots: Optional[Sequence[Path]] = None, maxdepth: int = 7,
              want: str = "any") -> Optional[Path]:
    """Breadth-first search for a file or directory called `name` under the
    Kaggle input root (mount layout differs between datasets, kernel outputs
    and Kaggle versions, so never hard-code /kaggle/input/<slug>)."""
    roots = [Path(r) for r in (roots or [INPUT_ROOT])]
    for root in roots:
        if not root.exists():
            continue
        frontier = [root]
        for _ in range(maxdepth + 1):
            nxt = []
            for d in frontier:
                try:
                    entries = list(os.scandir(d))
                except (PermissionError, FileNotFoundError, NotADirectoryError):
                    continue
                for e in entries:
                    if e.name == name:
                        if want == "any" or (want == "dir") == e.is_dir():
                            return Path(e.path)
                # do not descend into big leaf folders (image class dirs)
                subdirs = [e for e in entries if e.is_dir()]
                if len(entries) - len(subdirs) > 3000:
                    continue
                nxt.extend(Path(e.path) for e in subdirs)
            frontier = nxt
            if not frontier:
                break
    return None


def find_all(name: str, roots: Optional[Sequence[Path]] = None, maxdepth: int = 7) -> List[Path]:
    out = []
    for root in [Path(r) for r in (roots or [INPUT_ROOT])]:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            depth = len(Path(dirpath).relative_to(root).parts)
            if name in filenames or name in dirnames:
                out.append(Path(dirpath) / name)
            if depth >= maxdepth or len(filenames) > 3000:
                dirnames[:] = []
    return out


def env_report() -> Dict:
    info = {"kaggle": IS_KAGGLE, "python": sys.version.split()[0], "cpus": os.cpu_count()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # torch missing in a CPU-only notebook is fine
        info["torch"] = None
        info["gpus"] = []
    try:
        import shutil
        total, used, free = shutil.disk_usage(str(WORK))
        info["work_free_gb"] = round(free / 1e9, 1)
    except Exception:
        pass
    print(json.dumps(info, indent=1))
    return info


def n_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def resource_line() -> str:
    parts = []
    try:
        import psutil
        vm = psutil.virtual_memory()
        parts.append(f"RAM {vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB")
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip().replace("\n", " | ")
        if out:
            parts.append("GPU " + out)
    except Exception:
        pass
    try:
        import shutil
        u = shutil.disk_usage(str(WORK))
        parts.append(f"work disk used {(u.total-u.free)/1e9:.1f} GB")
    except Exception:
        pass
    return time.strftime("%H:%M:%S ") + " ; ".join(parts)


def pip_install(*pkgs: str, quiet: bool = True) -> bool:
    """Best-effort pip install (needs Internet ON in the Kaggle settings)."""
    cmd = [sys.executable, "-m", "pip", "install", "--no-input"] + (["-q"] if quiet else []) + list(pkgs)
    try:
        subprocess.run(cmd, check=True, timeout=900)
        return True
    except Exception as e:
        print(f"[pip] could not install {pkgs}: {e}")
        return False


# ----------------------------------------------------------------------------
# 2. seeds / progress / resume
# ----------------------------------------------------------------------------
def seed_everything(seed: int = 0, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    except Exception:
        pass


class Progress:
    """Tiny JSON ledger in /kaggle/working so a re-run (or a resumed version
    that attaches the previous output) skips finished units of work."""

    def __init__(self, path: Path | str = None, name: str = "progress.json"):
        self.path = Path(path) if path else WORK / name
        self.state = {}
        if self.path.exists():
            try:
                self.state = json.loads(self.path.read_text())
            except Exception:
                self.state = {}

    def done(self, key: str) -> bool:
        return self.state.get(key, {}).get("status") == "done"

    def mark(self, key: str, status: str = "done", **info) -> None:
        self.state[key] = {"status": status, "t": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str))
        tmp.replace(self.path)

    def seed_from(self, previous: Optional[Path]) -> None:
        """Merge a progress.json from an attached previous run output."""
        if previous and Path(previous).exists():
            prev = json.loads(Path(previous).read_text())
            for k, v in prev.items():
                self.state.setdefault(k, v)
            self.path.write_text(json.dumps(self.state, indent=1, default=str))


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t0
        print(f"[time] {self.label}: {self.dt/60:.1f} min")


# ----------------------------------------------------------------------------
# 3. child-wise splits
# ----------------------------------------------------------------------------
def make_child_splits(children: pd.DataFrame, label_col: str = "label", n_splits: int = 5,
                      seeds: Sequence[int] = (0, 1, 2), test_frac: float = 0.2,
                      external_site: Optional[str] = None, site_col: str = "site",
                      strat_extra: Sequence[str] = ("sex",), random_state: int = 2026) -> pd.DataFrame:
    """One row per child in -> adds `split` ('test'|'cv') and `fold_s{seed}`.

    * The locked test set is either an external site (preferred, plan 7) or a
      stratified 20 % of children.  It is chosen once, by child, and every
      notebook reads it from the manifest - nobody re-splits.
    * CV folds are stratified by label (+ sex when it is balanced enough) and
      are by construction grouped by child because there is one row per child.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    df = children.copy().reset_index(drop=True)
    if df["child_id"].duplicated().any():
        raise ValueError("make_child_splits expects one row per child")
    strat = df[label_col].astype(str)
    for c in strat_extra:
        if c in df and df[c].notna().all():
            cand = strat + "_" + df[c].astype(str)
            if cand.value_counts().min() >= max(2, n_splits):
                strat = cand
    df["split"] = "cv"
    if external_site is not None and site_col in df and (df[site_col] == external_site).any():
        df.loc[df[site_col] == external_site, "split"] = "test"
    elif test_frac and test_frac > 0:
        _, te = train_test_split(df.index, test_size=test_frac, stratify=strat, random_state=random_state)
        df.loc[te, "split"] = "test"
    cv_idx = df.index[df["split"] == "cv"]
    for s in seeds:
        col = f"fold_s{s}"
        df[col] = -1
        k = min(n_splits, int(df.loc[cv_idx, label_col].value_counts().min()))
        skf = StratifiedKFold(n_splits=max(2, k), shuffle=True, random_state=random_state + s)
        for f, (_, va) in enumerate(skf.split(cv_idx, strat.loc[cv_idx])):
            df.loc[cv_idx[va], col] = f
    return df


def fold_ids(children: pd.DataFrame, seed: int, fold: int):
    """(train_ids, val_ids, test_ids) for one seed/fold - child IDs only."""
    col = f"fold_s{seed}"
    cv = children[children["split"] == "cv"]
    tr = cv.loc[cv[col] != fold, "child_id"].tolist()
    va = cv.loc[cv[col] == fold, "child_id"].tolist()
    te = children.loc[children["split"] == "test", "child_id"].tolist()
    return tr, va, te


def assert_no_leak(*id_lists: Iterable) -> None:
    sets = [set(x) for x in id_lists]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = sets[i] & sets[j]
            if inter:
                raise AssertionError(f"child leakage between split {i} and {j}: {list(inter)[:5]}")


# ----------------------------------------------------------------------------
# 4. metrics / calibration / decision curves
# ----------------------------------------------------------------------------
def _safe_auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def _safe_ap(y, p):
    from sklearn.metrics import average_precision_score
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def threshold_youden(y, p) -> float:
    from sklearn.metrics import roc_curve
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    return float(np.clip(thr[j], 1e-6, 1 - 1e-6))


def threshold_for_sensitivity(y, p, target: float = 0.85) -> float:
    """Largest threshold that still reaches the target sensitivity on
    validation data (screening tools favour sensitivity)."""
    y = np.asarray(y)
    p = np.asarray(p)
    pos = np.sort(p[y == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[min(k, len(pos) - 1)] - 1e-9)


def binary_metrics(y, p, thr: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")
    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    ppv, npv = div(tp, tp + fp), div(tn, tn + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    return {
        "n": int(len(y)), "prev": float(y.mean()) if len(y) else float("nan"),
        "auroc": _safe_auc(y, p), "auprc": _safe_ap(y, p),
        "sens": sens, "spec": spec, "ppv": ppv, "npv": npv, "f1": f1,
        "bacc": np.nanmean([sens, spec]), "thr": float(thr),
        "brier": float(np.mean((p - y) ** 2)) if len(y) else float("nan"),
        "ece": expected_calibration_error(y, p),
    }


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_curve(y, p, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "p_mean": p[m].mean(), "y_mean": y[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class TemperatureScaler:
    """Binary temperature scaling fitted on validation probabilities."""

    def __init__(self):
        self.T = 1.0

    def fit(self, y, p):
        from scipy.optimize import minimize_scalar
        z = logit(p)
        y = np.asarray(y, dtype=float)

        def nll(logT):
            q = np.clip(sigmoid(z / np.exp(logT)), 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        if len(np.unique(y)) == 2:
            self.T = float(np.exp(minimize_scalar(nll, bounds=(np.log(0.25), np.log(10.0)), method="bounded").x))
        return self

    def transform(self, p):
        return sigmoid(logit(p) / self.T)


def decision_curve(y, p, thresholds=None) -> pd.DataFrame:
    """Net benefit of the model vs treat-all / treat-none (Vickers 2006)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    prev = y.mean()
    thresholds = np.linspace(0.05, 0.6, 23) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        yhat = p >= t
        tp = ((yhat == 1) & (y == 1)).sum()
        fp = ((yhat == 1) & (y == 0)).sum()
        w = t / (1 - t)
        rows.append({"thr": t, "nb_model": tp / n - fp / n * w,
                     "nb_all": prev - (1 - prev) * w, "nb_none": 0.0})
    return pd.DataFrame(rows)


def triage_metrics(y, p, uncertainty, thr: float, refer_q: float) -> Dict[str, float]:
    """Three-way output: refer the most uncertain fraction `refer_q`; report
    the refer rate and accuracy on the children that were not referred."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    u = np.asarray(uncertainty, dtype=float)
    cut = np.quantile(u, 1 - refer_q) if refer_q > 0 else np.inf
    keep = u < cut
    m = binary_metrics(y[keep], p[keep], thr) if keep.sum() > 1 else {}
    return {"refer_rate": float(1 - keep.mean()), **{f"kept_{k}": v for k, v in m.items()}}


# ----------------------------------------------------------------------------
# 5. statistics
# ----------------------------------------------------------------------------
def bootstrap_ci(y, p, fn: Callable = None, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap over children (rows are children)."""
    fn = fn or _safe_auc
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(fn(y[idx], p[idx]))
    stats = np.asarray(stats, dtype=float)
    est = fn(y, p)
    if len(stats) == 0:
        return {"est": est, "lo": float("nan"), "hi": float("nan")}
    return {"est": float(est), "lo": float(np.nanpercentile(stats, 100 * alpha / 2)),
            "hi": float(np.nanpercentile(stats, 100 * (1 - alpha / 2)))}


def bootstrap_metrics_ci(y, p, thr: float, n_boot: int = 2000, seed: int = 0,
                         keys=("auroc", "auprc", "sens", "spec", "ppv", "npv", "f1", "bacc", "brier", "ece")) -> pd.DataFrame:
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    base = binary_metrics(y, p, thr)
    boots = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = binary_metrics(y[idx], p[idx], thr)
        for k in keys:
            boots[k].append(m[k])
    rows = []
    for k in keys:
        b = np.asarray(boots[k], dtype=float)
        rows.append({"metric": k, "est": base[k],
                     "lo": np.nanpercentile(b, 2.5) if len(b) else np.nan,
                     "hi": np.nanpercentile(b, 97.5) if len(b) else np.nan})
    return pd.DataFrame(rows)


def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def delong_test(y, p1, p2) -> Dict[str, float]:
    """Paired DeLong test for two correlated AUROCs (Sun & Xu 2014)."""
    from scipy import stats
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")
    preds = np.vstack([np.asarray(p1, float), np.asarray(p2, float)])[:, order]
    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        return {"auc1": np.nan, "auc2": np.nan, "z": np.nan, "p": np.nan}
    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "z": float(z), "p": float(p)}


def mcnemar_test(y, yhat1, yhat2) -> Dict[str, float]:
    from scipy import stats
    y = np.asarray(y)
    c1 = np.asarray(yhat1) == y
    c2 = np.asarray(yhat2) == y
    b = int((c1 & ~c2).sum())
    c = int((~c1 & c2).sum())
    if b + c == 0:
        return {"b": b, "c": c, "p": 1.0}
    if b + c < 25:  # exact binomial
        p = float(min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5)))
    else:
        chi = (abs(b - c) - 1) ** 2 / (b + c)
        p = float(stats.chi2.sf(chi, 1))
    return {"b": b, "c": c, "p": p}


def holm(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ----------------------------------------------------------------------------
# 6. fairness
# ----------------------------------------------------------------------------
def age_band(age_months: pd.Series) -> pd.Series:
    return pd.cut(age_months, [0, 24, 36, 48, 1000], labels=["<24m", "24-35m", "36-47m", "48m+"], right=False)


def fairness_table(df: pd.DataFrame, group_cols: Sequence[str], thr: float,
                   y_col: str = "y_true", p_col: str = "prob") -> pd.DataFrame:
    rows = []
    for g in group_cols:
        if g not in df:
            continue
        for val, sub in df.groupby(g, observed=True):
            m = binary_metrics(sub[y_col], sub[p_col], thr)
            rows.append({"group": g, "value": str(val), **{k: m[k] for k in ("n", "prev", "auroc", "sens", "spec", "bacc", "ece")}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 7. perceptual hash duplicate audit (pure numpy DCT pHash)
# ----------------------------------------------------------------------------
def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)
    M = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    M[0] *= 1 / np.sqrt(2)
    return M * np.sqrt(2 / n)


_D32 = _dct_matrix(32)


def phash(img: np.ndarray) -> int:
    """64-bit pHash of an HxW or HxWx3 uint8 image."""
    import cv2
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = _D32 @ small @ _D32.T
    low = d[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return int(np.packbits(bits).view(">u8")[0])


def near_duplicates(hashes: Sequence[int], keys: Sequence[str], max_dist: int = 4,
                    max_pairs: int = 200000) -> pd.DataFrame:
    """All pairs with Hamming distance <= max_dist (bucketed on 16-bit bands,
    which is exact for max_dist <= 3 and near-exact for 4)."""
    h = np.asarray(hashes, dtype=np.uint64)
    buckets: Dict = {}
    for band in range(4):
        sub = (h >> np.uint64(16 * band)) & np.uint64(0xFFFF)
        for i, s in enumerate(sub.tolist()):
            buckets.setdefault((band, s), []).append(i)
    seen = set()
    rows = []
    for idxs in buckets.values():
        if len(idxs) < 2 or len(idxs) > 500:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                d = bin(int(h[i]) ^ int(h[j])).count("1")
                if d <= max_dist:
                    rows.append({"a": keys[i], "b": keys[j], "dist": d})
                    if len(rows) >= max_pairs:
                        return pd.DataFrame(rows)
    return pd.DataFrame(rows, columns=["a", "b", "dist"])


# ----------------------------------------------------------------------------
# 8. prediction files
# ----------------------------------------------------------------------------
PRED_COLS = ["child_id", "y_true", "prob", "uncertainty", "model", "seed", "fold", "part"]


def save_preds(df: pd.DataFrame, model: str, fold, out_dir: Path | str = None, seed: int = 0) -> Path:
    """Write preds_<model>_<fold>.csv (one row per child per part).
    part = 'val' (out-of-fold) or 'test' (locked test set)."""
    out_dir = Path(out_dir or WORK / "preds")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["model"] = model
    df["seed"] = seed
    df["fold"] = fold
    if "uncertainty" not in df:
        df["uncertainty"] = np.nan
    parts = "-".join(sorted(df["part"].astype(str).unique())) if "part" in df else "all"
    path = out_dir / f"preds_{model}_s{seed}_f{fold}_{parts}.csv"
    df[[c for c in PRED_COLS if c in df] + [c for c in df.columns if c not in PRED_COLS]].to_csv(path, index=False)
    return path


def load_all_preds(roots: Sequence[Path] = None) -> pd.DataFrame:
    roots = roots or [INPUT_ROOT, WORK]
    files = []
    for r in roots:
        r = Path(r)
        if r.exists():
            files += list(r.rglob("preds_*.csv"))
    seen, dfs = set(), []
    for f in files:
        if f.name in seen:  # same file mounted twice
            continue
        seen.add(f.name)
        try:
            d = pd.read_csv(f)
            d["src_file"] = f.name
            dfs.append(d)
        except Exception as e:
            print("skip", f, e)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=PRED_COLS)


def summarise_model(preds: pd.DataFrame, target_sens: float = 0.85, n_boot: int = 2000) -> Dict:
    """Pool OOF predictions over folds per seed, fix the threshold on OOF
    (validation) data, then evaluate the locked test set with that threshold.
    Test predictions from the K fold models are averaged per child."""
    out = {}
    val = preds[preds["part"] == "val"]
    test = preds[preds["part"] == "test"]
    if len(val):
        v = val.groupby(["seed", "child_id"], as_index=False).agg(y_true=("y_true", "first"), prob=("prob", "mean"))
        thr = float(np.median([threshold_for_sensitivity(g.y_true, g.prob, target_sens) for _, g in v.groupby("seed")]))
        vm = v.groupby("child_id", as_index=False).agg(y_true=("y_true", "first"), prob=("prob", "mean"))
        out["oof"] = binary_metrics(vm.y_true, vm.prob, thr)
        out["oof_ci"] = bootstrap_metrics_ci(vm.y_true.values, vm.prob.values, thr, n_boot=n_boot)
        out["thr"] = thr
    else:
        thr = 0.5
        out["thr"] = thr
    if len(test):
        t = test.groupby("child_id", as_index=False).agg(y_true=("y_true", "first"), prob=("prob", "mean"),
                                                        uncertainty=("uncertainty", "mean"))
        out["test"] = binary_metrics(t.y_true, t.prob, thr)
        out["test_ci"] = bootstrap_metrics_ci(t.y_true.values, t.prob.values, thr, n_boot=n_boot)
        out["test_df"] = t
    return out


# ----------------------------------------------------------------------------
# 9. parallel fold runner: one fold per GPU, folds queued
# ----------------------------------------------------------------------------
def run_jobs_parallel(module: str, func: str, jobs: List[Dict], log_dir: Path | str = None,
                      gpus: Optional[int] = None, extra_paths: Sequence[str] = (),
                      raise_on_error: bool = True) -> List[int]:
    """Run `module.func(job)` for every job in a separate Python process.
    With 2 T4s two folds train side by side (plan 8: faster than DDP for
    small models).  Each job must be JSON-serialisable and must skip itself
    if its outputs already exist (resume)."""
    log_dir = Path(log_dir or WORK / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    ng = n_gpus() if gpus is None else gpus
    slots = list(range(max(1, ng)))
    paths = [str(LIB_DIR)] + list(extra_paths)
    running: List = []
    codes = []
    pending = list(jobs)

    def launch(job, slot):
        env = os.environ.copy()
        if ng > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
        env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
        name = job.get("name", f"job{len(codes)+len(running)}")
        logf = open(log_dir / f"{name}.log", "w")
        code = (f"import json,sys; import {module} as M; "
                f"M.{func}(json.loads(sys.argv[1]))")
        p = subprocess.Popen([sys.executable, "-c", code, json.dumps(job)], stdout=logf, stderr=subprocess.STDOUT, env=env)
        return (p, slot, name, logf)

    free = list(slots)
    last_mon = 0.0
    while pending or running:
        if time.time() - last_mon > 300:  # resource monitor -> visible in the Kaggle log
            last_mon = time.time()
            print("[mon] " + resource_line(), flush=True)
        while pending and free:
            running.append(launch(pending.pop(0), free.pop(0)))
        time.sleep(5)
        still = []
        for p, slot, name, logf in running:
            rc = p.poll()
            if rc is None:
                still.append((p, slot, name, logf))
            else:
                logf.close()
                codes.append(rc)
                free.append(slot)
                tail = (log_dir / f"{name}.log").read_text(errors="ignore").splitlines()[-6:]
                print(f"[job] {name} exit={rc} :: " + " | ".join(tail))
        running = still
    bad = sum(c != 0 for c in codes)
    if bad and raise_on_error:
        raise RuntimeError(f"{bad}/{len(codes)} jobs failed - see {log_dir}")
    return codes


# ----------------------------------------------------------------------------
# 10. synthetic DEV cohort
# ----------------------------------------------------------------------------
@dataclass
class DevCohortSpec:
    n_children: int = 60
    clips_per_child: int = 6
    clip_seconds: float = 4.0
    fps: int = 10
    sr: int = 16000
    p_no_voice: float = 0.3
    p_no_cry: float = 0.5
    signal: float = 0.6          # strength of the injected label signal
    device_confound: bool = False  # True => device correlates with label (shortcut test must fire)
    seed: int = 7


def make_dev_cohort(out_dir: Path, face_images_happy: List[Path], face_images_neutral: List[Path],
                    spec: DevCohortSpec = DevCohortSpec()) -> Path:
    """Write a small synthetic cohort in exactly the format the real cohort
    must have (metadata/children.csv, clips.csv, cry.csv + mp4/wav files).

    Signal: 'ASD' children smile less often on social clips and have flatter
    synthetic pitch.  This exists only so every notebook can be executed end
    to end before ethics approval; it is not data about any child.
    """
    import cv2
    import soundfile as sf

    rng = np.random.default_rng(spec.seed)
    out_dir = Path(out_dir)
    (out_dir / "video").mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "cry").mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    stimuli = ["social_smile", "name_call", "toy", "peekaboo", "emotional_faces", "nonsocial_pattern",
               "bubbles", "song", "puppet", "face_to_face"]
    devices = ["tabA", "tabB"]
    rooms = ["room1", "room2"]

    def load_face(p):
        im = cv2.imread(str(p))
        return cv2.resize(im, (120, 120)) if im is not None else None

    happy = [im for im in (load_face(p) for p in face_images_happy[:400]) if im is not None]
    neutral = [im for im in (load_face(p) for p in face_images_neutral[:400]) if im is not None]
    if not happy or not neutral:
        raise RuntimeError("need face images to build the DEV cohort")

    children, clips, cries = [], [], []
    for c in range(spec.n_children):
        cid = f"DEV{c:04d}"
        group = rng.choice(["ASD", "TD", "DD"], p=[0.45, 0.4, 0.15])
        label = int(group == "ASD")
        age = int(rng.integers(18, 72))
        sex = rng.choice(["M", "F"], p=[0.7, 0.3] if label else [0.5, 0.5])
        device = (devices[label] if rng.random() < 0.85 else devices[1 - label]) if spec.device_confound else rng.choice(devices)
        room = rng.choice(rooms)
        smile_rate = (0.55 - spec.signal * 0.35 * label) + rng.normal(0, 0.08)
        f0_sd = (40 - spec.signal * 25 * label) + rng.normal(0, 5)
        mchat_items = (rng.random(20) < (0.15 + 0.35 * label * spec.signal)).astype(int)
        children.append({"child_id": cid, "label": label, "group": group, "age_months": age, "sex": sex,
                         "site": "dev_site", "device": device, "room": room, "recorder": rng.choice(["r1", "r2"]),
                         "mchat_total": int(mchat_items.sum()),
                         **{f"mchat_{i+1:02d}": int(v) for i, v in enumerate(mchat_items)}})
        h_img = happy[int(rng.integers(len(happy)))]
        n_img = neutral[int(rng.integers(len(neutral)))]
        bg = np.full((240, 320, 3), int(rng.integers(60, 200)), np.uint8)
        has_voice = rng.random() > spec.p_no_voice
        for k in range(spec.clips_per_child):
            stim = stimuli[k % len(stimuli)]
            vid = out_dir / "video" / f"{cid}_{k:02d}.mp4"
            nframes = int(spec.clip_seconds * spec.fps)
            vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), spec.fps, (320, 240))
            social = stim not in ("nonsocial_pattern", "toy")
            smiling = False
            x0, y0 = int(rng.integers(60, 140)), int(rng.integers(40, 80))
            for t in range(nframes):
                if t % spec.fps == 0:
                    smiling = rng.random() < (smile_rate if social else 0.2)
                frame = bg.copy()
                face = h_img if smiling else n_img
                dx, dy = int(rng.integers(-3, 4)), int(rng.integers(-3, 4))
                yy, xx = np.clip(y0 + dy, 0, 240 - 120), np.clip(x0 + dx, 0, 320 - 120)
                frame[yy:yy + 120, xx:xx + 120] = face
                vw.write(frame)
            vw.release()
            apath = ""
            if has_voice:
                apath = out_dir / "audio" / f"{cid}_{k:02d}.wav"
                t_ = np.arange(int(spec.sr * spec.clip_seconds)) / spec.sr
                f0 = 300 + f0_sd * np.sin(2 * np.pi * rng.uniform(0.5, 2) * t_) + rng.normal(0, 3, len(t_)).cumsum() * 0.01
                ph = 2 * np.pi * np.cumsum(f0) / spec.sr
                env = (np.sin(2 * np.pi * 3 * t_) > 0.2).astype(float)
                wav = 0.2 * env * (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)) + 0.02 * rng.normal(size=len(t_))
                sf.write(str(apath), wav.astype(np.float32), spec.sr)
            clips.append({"child_id": cid, "clip_id": f"{cid}_{k:02d}", "stimulus": stim,
                          "video_path": str(vid.relative_to(out_dir)),
                          "audio_path": str(Path(apath).relative_to(out_dir)) if apath else ""})
        if rng.random() > spec.p_no_cry:
            cpath = out_dir / "cry" / f"{cid}_cry.wav"
            t_ = np.arange(int(spec.sr * 3)) / spec.sr
            f0 = 450 + (60 - 30 * label * spec.signal) * np.sin(2 * np.pi * 1.5 * t_)
            ph = 2 * np.pi * np.cumsum(f0) / spec.sr
            wav = 0.3 * (np.sin(ph) + 0.6 * np.sin(2 * ph)) * (np.sin(2 * np.pi * 0.8 * t_) > -0.3) + 0.03 * rng.normal(size=len(t_))
            sf.write(str(cpath), wav.astype(np.float32), spec.sr)
            cries.append({"child_id": cid, "cry_id": f"{cid}_cry", "cry_path": str(cpath.relative_to(out_dir))})
    pd.DataFrame(children).to_csv(out_dir / "metadata" / "children.csv", index=False)
    pd.DataFrame(clips).to_csv(out_dir / "metadata" / "clips.csv", index=False)
    pd.DataFrame(cries, columns=["child_id", "cry_id", "cry_path"]).to_csv(out_dir / "metadata" / "cry.csv", index=False)
    (out_dir / "SYNTHETIC_README.txt").write_text(SYNTHETIC_BANNER + "\n")
    return out_dir


def locate_cohort() -> tuple:
    """Return (root, is_dev). Real cohort = an attached dataset containing
    metadata/children.csv; otherwise the DEV cohort written by NB00 (in its
    output, attached as a kernel source) or in /kaggle/working."""
    for cand in find_all("children.csv"):
        root = cand.parent.parent if cand.parent.name == "metadata" else cand.parent
        is_dev = (root / "SYNTHETIC_README.txt").exists()
        if not is_dev:
            return root, False
    for cand in find_all("children.csv") + list((WORK).rglob("children.csv")):
        root = cand.parent.parent if cand.parent.name == "metadata" else cand.parent
        if (root / "SYNTHETIC_README.txt").exists():
            return root, True
    return None, True


# ----------------------------------------------------------------------------
# misc helpers used by several notebooks
# ----------------------------------------------------------------------------
def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def table_md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return df.to_string(index=False)


def fmt_ci(d: Dict, k: str = "auroc") -> str:
    return f"{d.get('est', np.nan):.3f} [{d.get('lo', np.nan):.3f}, {d.get('hi', np.nan):.3f}]"


def load_image_rgb(path: str, size: int = 160) -> np.ndarray:
    """cv2 read -> RGB uint8 resized square (picklable for multiprocessing)."""
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_CUBIC)


def copy_tree_small(src: Path, dst: Path, patterns=("*.csv", "*.json", "*.png", "*.md")) -> None:
    import shutil
    for pat in patterns:
        for f in Path(src).rglob(pat):
            t = Path(dst) / f.relative_to(src)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, t)
