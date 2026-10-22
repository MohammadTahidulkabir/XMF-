"""
asd_xai.py - explanations that are *measured*, not just drawn (plan 6).

  pixel     : Grad-CAM / Grad-CAM++ / Score-CAM / Eigen-CAM (pytorch-grad-cam)
              + region energy in eyes/mouth, deletion & insertion AUC,
              cascading-randomisation sanity check, perturbation stability
  time      : XMF frame attention + Integrated Gradients over frames
  behaviour : SHAP on child-level behaviour features, TCAV-style concept
              sensitivity in the face-embedding space
  modality  : exact Shapley over modalities (asd_models.modality_shapley)
  case      : nearest neighbours in the fusion embedding
  counterf. : minimal single-feature change that flips a behaviour model
"""
from __future__ import annotations

import copy
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

import asd_models as M


# ---------------------------------------------------------------- pixel level
def cam_methods():
    from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, ScoreCAM
    return {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "scorecam": ScoreCAM, "eigencam": EigenCAM}


def target_layer(model):
    for name in ("conv_head", "head_conv"):
        if hasattr(model, name):
            return getattr(model, name)
    if hasattr(model, "blocks"):
        return model.blocks[-1]
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    return list(model.children())[-3]


def compute_cams(model, x: torch.Tensor, method: str = "gradcam") -> np.ndarray:
    """x [B,3,H,W] normalised -> maps [B,H,W] in [0,1] for the ASD logit."""
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    cls = cam_methods()[method]
    with cls(model=model, target_layers=[target_layer(model)]) as cam:
        kw = {"batch_size": 16} if method == "scorecam" else {}
        try:
            out = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)] * x.shape[0], **kw)
        except TypeError:
            out = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)] * x.shape[0])
    return np.asarray(out, np.float32)


def region_energy(cams: np.ndarray, boxes: np.ndarray) -> Dict[str, float]:
    """Share of CAM energy inside eye+mouth boxes vs their share of the area.
    boxes [B,3,4] (x0,y0,x1,y1) in crop pixels, NaN when unknown."""
    inside, area = [], []
    H, W = cams.shape[1:]
    for c, bx in zip(cams, boxes):
        if np.isnan(bx).all():
            continue
        m = np.zeros((H, W), bool)
        for x0, y0, x1, y1 in bx:
            if np.isnan(x0):
                continue
            m[int(max(0, y0 - 4)):int(min(H, y1 + 4)), int(max(0, x0 - 4)):int(min(W, x1 + 4))] = True
        e = c.sum() + 1e-8
        inside.append(c[m].sum() / e)
        area.append(m.mean())
    return {"energy_in_eyes_mouth": float(np.mean(inside)) if inside else np.nan,
            "area_eyes_mouth": float(np.mean(area)) if area else np.nan,
            "energy_background": float(1 - np.mean(inside)) if inside else np.nan}


@torch.no_grad()
def deletion_insertion(model, x: torch.Tensor, cams: np.ndarray, steps: int = 20) -> Dict[str, float]:
    """Deletion: remove most-salient pixels first (-> prob should drop fast,
    low AUC is good).  Insertion: start from blurred image, add salient pixels
    (high AUC is good).  Probabilities of the ASD logit."""
    B, _, H, W = x.shape
    blur = F.avg_pool2d(x, 11, 1, 5)
    order = np.argsort(-cams.reshape(B, -1), axis=1)
    dels, ins = np.zeros((B, steps + 1)), np.zeros((B, steps + 1))
    for k in range(steps + 1):
        n = int(k / steps * H * W)
        mask = np.zeros((B, H * W), np.float32)
        for b in range(B):
            mask[b, order[b, :n]] = 1
        m = torch.tensor(mask.reshape(B, 1, H, W), device=x.device)
        dels[:, k] = torch.sigmoid(model(x * (1 - m) + blur * m).float().squeeze(1)).cpu().numpy()
        ins[:, k] = torch.sigmoid(model(blur * (1 - m) + x * m).float().squeeze(1)).cpu().numpy()
    trap = getattr(np, "trapezoid", None) or np.trapz
    auc = lambda a: trap(a, dx=1 / steps, axis=1).mean()
    return {"deletion_auc": float(auc(dels)), "insertion_auc": float(auc(ins))}


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmean([spearmanr(x.ravel(), y.ravel()).correlation for x, y in zip(a, b)]))


def sanity_cascade(model, x, method="gradcam", n_layers: int = 3) -> List[Dict]:
    """Adebayo et al. 2018: randomise weights from the top layer down; maps
    must change (low rank correlation) or the explanation ignores the model."""
    base = compute_cams(model, x, method)
    out = []
    m = copy.deepcopy(model)
    mods = [mm for mm in m.modules() if isinstance(mm, (torch.nn.Conv2d, torch.nn.Linear))][::-1]
    for i, layer in enumerate(mods[:n_layers]):
        torch.nn.init.normal_(layer.weight, 0, 0.05)
        out.append({"randomised_top_layers": i + 1, "spearman_to_original": _rank_corr(base, compute_cams(m, x, method))})
    return out


def stability(model, x, method="gradcam") -> Dict[str, float]:
    base = compute_cams(model, x, method)
    bright = compute_cams(model, x + 0.2, method)
    shift = compute_cams(model, torch.roll(x, shifts=(4, 4), dims=(2, 3)), method)
    shift = np.roll(shift, shift=(-4, -4), axis=(1, 2))
    return {"spearman_brightness": _rank_corr(base, bright), "spearman_shift4px": _rank_corr(base, shift)}


# ----------------------------------------------------------------- time level
def integrated_gradients_frames(model, b: Dict, steps: int = 32) -> np.ndarray:
    """IG of P(ASD) w.r.t. the face input -> per-frame attribution [B,C,T]
    (zero baseline = 'average frame' after z-scoring)."""
    face = b["face"].detach()
    total = torch.zeros_like(face)
    model.eval()
    for a in torch.linspace(1.0 / steps, 1.0, steps, device=face.device):
        f = (face * a).requires_grad_(True)
        p = model({**b, "face": f})["prob"].sum()
        g, = torch.autograd.grad(p, f)
        total += g
    attr = (face * total / steps).sum(-1)
    return attr.detach().cpu().numpy()


# ------------------------------------------------------------ concept (TCAV)
def concept_vector(emb_pos: np.ndarray, emb_neg: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    X = np.r_[emb_pos, emb_neg]
    y = np.r_[np.ones(len(emb_pos)), np.zeros(len(emb_neg))]
    clf = LogisticRegression(C=0.1, max_iter=2000, random_state=seed).fit(X, y)
    v = clf.coef_[0]
    return v / (np.linalg.norm(v) + 1e-8)


def tcav_score(model, b: Dict, cav: np.ndarray, emb_dims: int) -> np.ndarray:
    """Directional derivative of P(ASD) along the concept direction in the
    (z-scored) embedding part of every valid frame; per-child mean -> sign."""
    face = b["face"].detach().clone().requires_grad_(True)
    p = model({**b, "face": face})["prob"].sum()
    g, = torch.autograd.grad(p, face)
    v = torch.zeros(face.shape[-1], device=face.device)
    v[:emb_dims] = torch.tensor(cav, dtype=torch.float32, device=face.device)
    d = (g * v).sum(-1)  # B,C,T
    fm = b["face_frame_mask"].float()
    return ((d * fm).sum((1, 2)) / fm.sum((1, 2)).clamp_min(1)).cpu().numpy()


# ------------------------------------------------------------- counterfactual
def single_feature_counterfactual(clf, x: np.ndarray, names: Sequence[str], thr: float,
                                  max_sd: float = 3.0, step: float = 0.05):
    """For a standardised linear/probabilistic model: smallest |delta| (in SD)
    on one feature that moves P(ASD) below thr.  Returns (feature, delta)."""
    best = (None, np.inf)
    p0 = clf.predict_proba(x[None])[0, 1]
    if p0 < thr:
        return best
    for j, n in enumerate(names):
        for sgn in (-1, 1):
            for d in np.arange(step, max_sd + 1e-9, step):
                z = x.copy()
                z[j] += sgn * d
                if clf.predict_proba(z[None])[0, 1] < thr:
                    if d < abs(best[1]):
                        best = (n, sgn * d)
                    break
    return best
