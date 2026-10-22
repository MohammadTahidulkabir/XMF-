"""
asd_face.py - face pipeline for protocol videos (NB01).

Video -> frames at a fixed rate -> MediaPipe Face Landmarker (478 landmarks,
52 blendshapes, head-pose matrix) -> eye-aligned square crop -> per-frame
handcrafted behaviour features:

  52 blendshapes (action-unit proxies: smile, blink, brow raise, jaw open ...)
  head pose yaw/pitch/roll (deg), gaze proxy (iris offset, both eyes),
  eye-aspect ratio (both eyes), face size + centre, detected flag  -> 64 dims

Also stored for later notebooks: region boxes (eyes, mouth) in crop
coordinates for XAI localisation, frame-quality stats and face-masked
background thumbnails for the shortcut tests (NB09).

MediaPipe runs on CPU; we use all cores with one landmarker per process,
so NB01 costs no GPU quota.
"""
from __future__ import annotations

import math
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/latest/face_landmarker.task")
BLENDSHAPES = 52
HAND_DIM = BLENDSHAPES + 12
HAND_NAMES: List[str] = []  # filled on first detection (blendshape names)
EXTRA_NAMES = ["yaw", "pitch", "roll", "gaze_lx", "gaze_ly", "gaze_rx", "gaze_ry", "ear_l", "ear_r",
               "face_size", "face_cx", "face_cy"]
# MediaPipe mesh indices
L_EYE = [33, 160, 158, 133, 153, 144]
R_EYE = [362, 385, 387, 263, 373, 380]
L_IRIS, R_IRIS = 468, 473
MOUTH = [61, 291, 0, 17, 13, 14, 78, 308]


def get_landmarker_model(dst: Path) -> Optional[Path]:
    dst = Path(dst)
    if dst.exists():
        return dst
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LANDMARKER_URL, dst)
        return dst
    except Exception as e:
        print("[face] could not download landmarker:", e)
        return None


class FaceAnalyzer:
    def __init__(self, model_path: Optional[Path], crop: int = 160):
        import cv2
        self.cv2 = cv2
        self.crop = crop
        self.lm = None
        if model_path is not None:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python as mpt
                from mediapipe.tasks.python import vision
                opts = vision.FaceLandmarkerOptions(
                    base_options=mpt.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=vision.RunningMode.IMAGE, num_faces=1,
                    output_face_blendshapes=True, output_facial_transformation_matrixes=True,
                    min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
                self.lm = vision.FaceLandmarker.create_from_options(opts)
                self.mp = mp
            except Exception as e:
                print("[face] mediapipe unavailable -> Haar fallback:", e)
        if self.lm is None:
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # ---- geometry helpers
    def _align(self, rgb, le, re):
        """Similarity transform: eyes at (0.35,0.40) and (0.65,0.40) of crop."""
        S = self.crop
        dst_l, dst_r = np.array([0.35 * S, 0.40 * S]), np.array([0.65 * S, 0.40 * S])
        v_src, v_dst = re - le, dst_r - dst_l
        scale = np.linalg.norm(v_dst) / max(1e-6, np.linalg.norm(v_src))
        ang = math.atan2(v_src[1], v_src[0]) - math.atan2(v_dst[1], v_dst[0])
        c, s = math.cos(-ang) * scale, math.sin(-ang) * scale
        M = np.array([[c, -s, 0], [s, c, 0]], dtype=np.float64)
        M[:, 2] = dst_l - M[:, :2] @ le
        out = self.cv2.warpAffine(rgb, M, (S, S), flags=self.cv2.INTER_LINEAR, borderMode=self.cv2.BORDER_REPLICATE)
        return out, M

    @staticmethod
    def _ear(p):
        a = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
        return a / (2 * np.linalg.norm(p[0] - p[3]) + 1e-6)

    @staticmethod
    def _euler(R):
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        yaw = math.degrees(math.atan2(-R[2, 0], sy))
        roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        return yaw, pitch, roll

    def analyse(self, rgb: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """-> (aligned crop | None, hand features [HAND_DIM], region boxes [3,4] | None, face bbox | None)"""
        H, W = rgb.shape[:2]
        feat = np.full(HAND_DIM, np.nan, np.float32)
        if self.lm is not None:
            res = self.lm.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            if not res.face_landmarks:
                return None, feat, None, None
            pts = np.array([[p.x * W, p.y * H] for p in res.face_landmarks[0]], dtype=np.float64)
            if res.face_blendshapes:
                bs = res.face_blendshapes[0]
                if not HAND_NAMES:
                    HAND_NAMES.extend([b.category_name for b in bs] + EXTRA_NAMES)
                feat[:BLENDSHAPES] = [b.score for b in bs][:BLENDSHAPES]
            if res.facial_transformation_matrixes:
                feat[BLENDSHAPES:BLENDSHAPES + 3] = self._euler(np.asarray(res.facial_transformation_matrixes[0])[:3, :3])
            le, re = pts[L_EYE].mean(0), pts[R_EYE].mean(0)
            if len(pts) > R_IRIS:
                for k, (iris, eye) in enumerate(((L_IRIS, L_EYE), (R_IRIS, R_EYE))):
                    e = pts[eye]
                    w = np.linalg.norm(e[3] - e[0]) + 1e-6
                    off = (pts[iris] - e.mean(0)) / w
                    feat[BLENDSHAPES + 3 + 2 * k: BLENDSHAPES + 5 + 2 * k] = off
            feat[BLENDSHAPES + 7] = self._ear(pts[L_EYE])
            feat[BLENDSHAPES + 8] = self._ear(pts[R_EYE])
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
        else:
            gray = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2GRAY)
            faces = self.haar.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
            if len(faces) == 0:
                return None, feat, None, None
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            x0, y0, x1, y1 = x, y, x + w, y + h
            le = np.array([x + 0.32 * w, y + 0.40 * h])
            re = np.array([x + 0.68 * w, y + 0.40 * h])
            pts = None
        feat[BLENDSHAPES + 9] = (x1 - x0) / W
        feat[BLENDSHAPES + 10] = (x0 + x1) / 2 / W
        feat[BLENDSHAPES + 11] = (y0 + y1) / 2 / H
        crop, M = self._align(rgb, le, re)
        boxes = None
        if pts is not None:
            P = (M[:, :2] @ pts.T).T + M[:, 2]
            boxes = np.stack([np.r_[P[ix].min(0), P[ix].max(0)] for ix in (L_EYE, R_EYE, MOUTH)]).astype(np.float32)
        return crop, feat, boxes, np.array([x0, y0, x1, y1], np.float32)


def read_frames(path: str, fps: float, max_frames: int):
    """Decode a video once at a fixed sampling rate -> list of RGB frames."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1.0, src_fps / fps)
    frames, i, next_t = [], 0, 0.0
    while len(frames) < max_frames:
        ok = cap.grab()
        if not ok:
            break
        if i >= next_t:
            ok, fr = cap.retrieve()
            if ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            next_t += step
        i += 1
        if n and i >= n:
            break
    cap.release()
    return frames


def quality_stats(rgb: np.ndarray) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.array([g.mean() / 255, g.std() / 255, math.log1p(cv2.Laplacian(g, cv2.CV_64F).var())], np.float32)


def background_thumb(rgb: np.ndarray, bbox: Optional[np.ndarray], size: int = 112) -> np.ndarray:
    import cv2
    im = rgb.copy()
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 1.4, (y1 - y0) * 1.6
        im[int(max(0, cy - h / 2)):int(cy + h / 2), int(max(0, cx - w / 2)):int(cx + w / 2)] = 0
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)


_ANALYZER = None


def _init_worker(model_path, crop):
    global _ANALYZER
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _ANALYZER = FaceAnalyzer(model_path, crop)


def process_child(args) -> Dict:
    """Worker: all clips of one child -> files under out_root."""
    cid, clip_rows, cohort_root, out_root, fps, max_t, crop, n_bg = args
    out_root = Path(out_root)
    done_rows = out_root / "rows" / f"{cid}.json"
    if (out_root / "crops" / f"{cid}.npy").exists() and done_rows.exists():
        import json
        return {"child_id": cid, "rows": json.loads(done_rows.read_text()), "skipped": True, "names": list(HAND_NAMES)}
    import asd_utils as U
    C = len(clip_rows)
    crops = np.zeros((C, max_t, crop, crop, 3), np.uint8)
    valid = np.zeros((C, max_t), bool)
    hand = np.full((C, max_t, HAND_DIM), np.nan, np.float16)
    boxes = np.full((C, max_t, 3, 4), np.nan, np.float16)
    qual = np.zeros((C, max_t, 3), np.float16)
    bg = np.zeros((C, n_bg, 112, 112, 3), np.uint8)
    rows = []
    for c, r in enumerate(clip_rows):
        frames = read_frames(Path(cohort_root) / r["video_path"], fps, max_t)
        bb_last = None
        for t, fr in enumerate(frames):
            cr, f, bx, bb = _ANALYZER.analyse(fr)
            qual[c, t] = quality_stats(fr)
            hand[c, t] = f
            if cr is not None:
                crops[c, t], valid[c, t] = cr, True
                bb_last = bb
                if bx is not None:
                    boxes[c, t] = bx
        if frames:
            for k, t in enumerate(np.linspace(0, len(frames) - 1, n_bg).round().astype(int)):
                bg[c, k] = background_thumb(frames[t], bb_last)
        ph = U.phash(crops[c, np.flatnonzero(valid[c])[0]]) if valid[c].any() else 0
        rows.append({"child_id": cid, "clip_idx": c, "clip_id": r["clip_id"], "stimulus": r.get("stimulus", ""),
                     "n_frames": len(frames), "n_valid": int(valid[c].sum()), "phash": str(ph)})
    for name, arr in (("valid", valid), ("hand", hand), ("boxes", boxes), ("qual", qual), ("bg", bg), ("crops", crops)):
        (out_root / name).mkdir(parents=True, exist_ok=True)
        np.save(out_root / name / f"{cid}.npy", arr)
    import json
    done_rows.parent.mkdir(parents=True, exist_ok=True)
    done_rows.write_text(json.dumps(rows))  # written last = child finished
    return {"child_id": cid, "rows": rows, "names": list(HAND_NAMES)}
