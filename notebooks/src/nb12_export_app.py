# %% [meta]
# {"slug": "asd-xmf-nb12-export-app", "title": "ASD-XMF NB12 Export App",
#  "gpu": false, "internet": true, "datasets": [],
#  "kernels": ["asd-xmf-nb01-face-pipeline", "asd-xmf-p1-photo-benchmark", "asd-xmf-nb08-fusion"], "phase": "B"}

# %% [markdown]
# # NB12 - Export for the offline clinician app
# * frame encoder (NB03 EfficientNet-B0 expression net, 256-d) and the XMF fusion model ->
#   ONNX (opset 17) -> dynamic **INT8** quantisation (ONNX Runtime)
# * numerical parity FP32 vs INT8, parameter count, FLOPs, file size, CPU latency
#   (Kaggle CPU is only a proxy - re-measure on a mid-range Android phone with ONNX Runtime
#   Mobile; MediaPipe runs face landmarks on-device)
# * optional TFLite via onnx2tf (skipped gracefully if it does not install)

# %%
import numpy as np, pandas as pd, torch, time
from pathlib import Path
import asd_models as M
U.pip_install("onnx", "onnxruntime", "onnxscript", "onnxconverter-common")
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat, CalibrationDataReader
OUT = U.WORK / "export"; OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
report = {}

def flops(model, *args):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            model(*args)
        return int(fc.get_total_flops())
    except Exception as e:
        return None

def onnx_export(model, args, path, **kw):
    """Legacy TorchScript exporter (robust for dict-input models); falls back on older torch."""
    try:
        torch.onnx.export(model, args, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, args, path, **kw)

def bench(path, feeds, n=30):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for _ in range(3): s.run(None, feeds)
    t = time.perf_counter()
    for _ in range(n): out = s.run(None, feeds)
    return (time.perf_counter() - t) / n * 1000, out

# %% [markdown]
# ## 1. Frame encoder (one face crop -> 256-d expression embedding + 7 expression logits)

# %%
ck = U.find_path("efficientnet_b0_expr_backbone.pt")
if ck:
    net = M.load_expr_net("efficientnet_b0", ck.parent).eval()
    class Enc(torch.nn.Module):
        def __init__(s, n): super().__init__(); s.n = n
        def forward(s, x): e = s.n.embed(x); return e, s.n.cls_head(e)
    enc = Enc(net).eval()
    x = torch.randn(1, 3, 160, 160)
    p32, p8 = OUT / "frame_encoder_fp32.onnx", OUT / "frame_encoder_int8.onnx"
    onnx_export(enc, x, p32, input_names=["image"], output_names=["embedding", "expr_logits"], opset_version=17,
                      dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}, "expr_logits": {0: "batch"}})
    # CNNs need *static* INT8 (activations calibrated on real face crops); dynamic INT8 only suits Linear layers
    calib = []
    fi = U.find_path("face_index.csv")
    if fi is not None:
        st = M.FaceStore(fi.parent)
        for cid in st.index.child_id.unique()[:16]:
            v = st.valid(cid); cr = st.crops(cid)
            for c in range(v.shape[0]):
                for t in np.flatnonzero(v[c])[::8][:2]:
                    calib.append(M.gpu_prepare(torch.from_numpy(np.asarray(cr[c, t]))[None], 160, train=False).float().numpy())
    class Reader(CalibrationDataReader):
        def __init__(s, xs): s.it = iter({"image": a} for a in xs)
        def get_next(s): return next(s.it, None)
    cal, held = calib[: max(1, len(calib) * 2 // 3)], calib[len(calib) * 2 // 3:]
    def fidelity(path):
        a, b = ort.InferenceSession(str(p32)), ort.InferenceSession(str(path))
        cs = [float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
              for u, v in ((a.run(None, {"image": h})[0][0], b.run(None, {"image": h})[0][0]) for h in held)]
        return float(np.mean(cs)) if cs else float("nan")
    chosen, cos8 = p32, float("nan")
    if len(calib) >= 8:
        quantize_static(str(p32), str(p8), Reader(cal), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        cos8 = fidelity(p8)
        chosen = p8
    if not (cos8 >= 0.98):  # EfficientNet (SiLU/SE/depthwise) usually fails INT8 -> ship FP16 instead
        import onnx
        from onnxconverter_common import float16
        p16 = OUT / "frame_encoder_fp16.onnx"
        onnx.save(float16.convert_float_to_float16(onnx.load(str(p32)), keep_io_types=True, op_block_list=["Gemm"]), str(p16))
        chosen = p16
    print(f"INT8 held-out embedding cosine = {cos8:.3f} -> shipping {chosen.name}")
    xr = calib[-1] if calib else x.numpy()
    ms32, o32 = bench(p32, {"image": xr}); ms8, o8 = bench(chosen, {"image": xr})
    report["frame_encoder"] = {"params": M.count_params(enc), "flops": flops(enc, x),
                               "mb_fp32": p32.stat().st_size / 1e6, "shipped": chosen.name, "mb_shipped": chosen.stat().st_size / 1e6,
                               "ms_fp32_cpu": ms32, "ms_shipped_cpu": ms8, "calibration_frames": len(calib), "int8_heldout_cosine": cos8,
                               "emb_cosine_fp32_vs_shipped": float(np.dot(o32[0][0], o8[0][0]) / (np.linalg.norm(o32[0][0]) * np.linalg.norm(o8[0][0])))}
    print(report["frame_encoder"])

# %% [markdown]
# ## 2. XMF fusion (fixed shapes: 10 clips x 40 frames; masks mark missing modalities)

# %%
cks = sorted(f for d in U.find_all("ckpt") for f in d.glob("xmf_noquest_s0_f*.pt")) or sorted(f for d in U.find_all("ckpt") for f in d.glob("xmf_s0_f*.pt"))
if cks:
    model, info = M.load_fusion_ckpt(cks[0], torch.device("cpu"))
    mods = model.mods
    C, T = 10, 40
    dims = info["dims"]
    ex = {"face": torch.randn(1, C, T, dims["face"]), "face_frame_mask": torch.ones(1, C, T, dtype=torch.bool),
          "face_mask": torch.ones(1, C, dtype=torch.bool)}
    for m in ("voice", "cry", "quest"):
        if m in mods:
            L = {"voice": C, "cry": 3}.get(m)
            ex[m] = torch.randn(1, L, dims[m]) if L else torch.randn(1, dims[m])
            ex[f"{m}_mask"] = torch.ones(1, L, dtype=torch.bool) if L else torch.ones(1, dtype=torch.bool)
    keys = list(ex.keys())
    class Wrap(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, *args):
            o = s.m(dict(zip(keys, args)))
            return o["prob"], o["unc"], o["gates"]
    w = Wrap(model).eval()
    args = tuple(ex[k] for k in keys)
    f32, f8 = OUT / "xmf_fusion_fp32.onnx", OUT / "xmf_fusion_int8.onnx"
    onnx_export(w, args, f32, input_names=keys, output_names=["p_asd", "uncertainty", "modality_gates"], opset_version=17)
    quantize_dynamic(str(f32), str(f8), weight_type=QuantType.QInt8)
    feeds = {k: v.numpy() for k, v in ex.items()}
    ms32, o32 = bench(f32, feeds); ms8, o8 = bench(f8, feeds)
    with torch.no_grad():
        pt = w(*args)[0].item()
    report["xmf_fusion"] = {"checkpoint": cks[0].name, "modalities": mods, "params": M.count_params(model), "flops": flops(w, *args),
                            "mb_fp32": f32.stat().st_size / 1e6, "mb_int8": f8.stat().st_size / 1e6,
                            "ms_fp32_cpu": ms32, "ms_int8_cpu": ms8, "p_torch": pt, "p_onnx_fp32": float(o32[0][0]), "p_onnx_int8": float(o8[0][0]),
                            "temperature": info["T"], "norm_stats_file": "xmf_norm.json"}
    U.save_json({m: {"mean": mu.tolist(), "sd": sd.tolist()} for m, (mu, sd) in info["norm"].items()} | {"temperature": info["T"], "inputs": keys},
                OUT / "xmf_norm.json")
    print(report["xmf_fusion"])

# %% [markdown]
# ## 3. Optional TFLite

# %%
if not PREFLIGHT and (OUT / "frame_encoder_fp32.onnx").exists():
    try:
        if U.pip_install("onnx2tf", "tensorflow", "onnx_graphsurgeon", "sng4onnx"):
            import subprocess
            r = subprocess.run(["onnx2tf", "-i", str(OUT / "frame_encoder_fp32.onnx"), "-o", str(OUT / "tflite"), "-oiqt"], capture_output=True, text=True, timeout=1800)
            print(r.stdout[-1500:], r.stderr[-1500:])
            report["tflite"] = sorted(p.name for p in (OUT / "tflite").glob("*.tflite"))
    except Exception as e:
        report["tflite"] = f"skipped: {e}"
U.save_json(report, OUT / "export_report.json")
print(json.dumps(report, indent=1, default=str))
