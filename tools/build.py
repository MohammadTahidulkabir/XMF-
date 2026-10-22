"""
Build Kaggle notebooks from the percent-format sources in notebooks/src.

    python tools/build.py            # -> kaggle_build/<slug>/{<slug>.ipynb, kernel-metadata.json}
    python tools/build.py --profile preflight   # CPU smoke-test variant (GPU off)

Each source starts with a `# %% [meta]` cell holding JSON (slug, title, gpu,
internet, datasets, kernels).  A shared bootstrap cell is injected after the
title so every notebook finds asd_lib the same way.
"""
import argparse
import json
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "src"
OUT = ROOT / "kaggle_build"
import os
def _user():
    try:
        return json.loads((Path(os.environ.get("ASD_KAGGLE_DIR", "D:/kaggle_project")) / "account.json").read_text())["user"]
    except Exception:
        return "tahidulkabir"
USER = os.environ.get("ASD_KAGGLE_USER") or _user()
LIB_DATASET = f"{USER}/asd-xmf-lib"

BOOTSTRAP = '''RUN_PROFILE = "{profile}"   # "full" on GPU; "preflight" = quick CPU smoke test (tools/push.py)
PREFLIGHT = RUN_PROFILE == "preflight"
import os, sys, glob, json, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _find_lib():
    for c in ("/kaggle/input/asd-xmf-lib", "/kaggle/input/datasets/{user}/asd-xmf-lib"):
        if os.path.exists(os.path.join(c, "asd_utils.py")):
            return c
    for depth in range(1, 6):
        hits = glob.glob("/kaggle/input/" + "*/" * depth + "asd_utils.py")
        if hits:
            return os.path.dirname(hits[0])
    return os.path.abspath(os.path.join(os.getcwd(), "..", "..", "asd_lib"))  # local run

def _require_internet():
    import socket
    try:
        socket.create_connection(("pypi.org", 443), timeout=10).close()
    except OSError:
        raise SystemExit("NO INTERNET in this Kaggle session: verify the account's phone number (Kaggle > Settings) "
                         "and keep 'Internet on'. Stopping so no step runs with missing models or features.")
_require_internet()

LIB = _find_lib()
sys.path.insert(0, LIB)
import asd_utils as U
print("asd_lib:", LIB, "| profile:", RUN_PROFILE)
ENV = U.env_report()
open(os.path.join(str(U.WORK), "_run_profile_" + RUN_PROFILE + ".txt"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))  # queue marker'''


def split_cells(text: str):
    cells, cur, kind = [], [], "code"
    for line in text.splitlines():
        m = re.match(r"^# %%(?: \[(\w+)\])?\s*$", line)
        if m:
            if cur:
                cells.append((kind, cur))
            kind, cur = (m.group(1) or "code"), []
            continue
        cur.append(line)
    if cur:
        cells.append((kind, cur))
    return cells


def build_one(src: Path, profile: str, variant: dict = None):
    cells = split_cells(src.read_text(encoding="utf-8"))
    meta = None
    nb = nbformat.v4.new_notebook()
    injected = False
    for kind, lines in cells:
        if kind == "meta":
            meta = json.loads("\n".join(l[1:].strip() if l.startswith("#") else l for l in lines))
            continue
        body = "\n".join(lines).strip("\n")
        if not body.strip():
            continue
        if kind == "markdown":
            md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            nb.cells.append(nbformat.v4.new_markdown_cell(md))
        else:
            if not injected:
                boot = BOOTSTRAP.format(profile=profile, user=USER)
                if variant:
                    boot += "\nVARIANT = " + json.dumps(variant.get("vars", {}))
                nb.cells.append(nbformat.v4.new_code_cell(boot))
                injected = True
            nb.cells.append(nbformat.v4.new_code_cell(body))
    if not injected:
        nb.cells.insert(1, nbformat.v4.new_code_cell(BOOTSTRAP.format(profile=profile, user=USER)))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    if variant:
        meta = {**meta, "slug": meta["slug"] + "-" + variant["suffix"], "title": variant["title"]}
    slug = meta["slug"]
    d = (OUT if profile == "full" else OUT.parent / f"kaggle_build_{profile}") / slug
    d.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, d / f"{slug}.ipynb")
    gpu = bool(meta.get("gpu")) and profile in ("full", "smoke")
    km = {
        "id": f"{USER}/{slug}",
        "title": meta["title"],
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_tpu": False,
        "enable_internet": bool(meta.get("internet", True)),
        "dataset_sources": [LIB_DATASET] + [d.replace("tahidulkabir/", USER + "/") for d in meta.get("datasets", [])],
        "kernel_sources": [f"{USER}/{k}" if "/" not in k else k for k in meta.get("kernels", [])],
        "competition_sources": [],
        "model_sources": meta.get("models", []),
    }
    if gpu:
        km["machine_shape"] = "NvidiaTeslaT4"  # T4 x2 on the free tier
    (d / "kernel-metadata.json").write_text(json.dumps(km, indent=2))
    return slug, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["full", "preflight", "smoke"])
    ap.add_argument("--only", default=None, help="substring filter on source file names")
    a = ap.parse_args()
    built = []
    for src in sorted(SRC.glob("*.py")):
        if a.only and a.only not in src.name:
            continue
        cells = split_cells(src.read_text(encoding="utf-8"))
        raw_meta = json.loads("\n".join(l[1:].strip() if l.startswith("#") else l for l in cells[0][1]))
        for variant in raw_meta.get("variants", [None]):
            slug, meta = build_one(src, a.profile, variant)
            built.append({"slug": slug, "src": src.name, "gpu": meta.get("gpu"), "phase": meta.get("phase"),
                          "kernels": meta.get("kernels", []), "order": meta.get("order", 99)})
            print(f"built {slug:55s} gpu={meta.get('gpu')} profile={a.profile}")
    OUT.mkdir(exist_ok=True)
    (OUT / f"build_{a.profile}.json").write_text(json.dumps(built, indent=1))


if __name__ == "__main__":
    main()
