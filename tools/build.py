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


def _read_meta(src: Path) -> dict:
    cells = split_cells(src.read_text(encoding="utf-8"))
    return json.loads("\n".join(l[1:].strip() if l.startswith("#") else l for l in cells[0][1]))


def section_code(src: Path) -> str:
    """The notebook source without its meta cell; markdown cells stay as comments."""
    out = []
    for kind, lines in split_cells(src.read_text(encoding="utf-8")):
        if kind == "meta":
            continue
        if kind == "markdown":
            lines = [l if l.startswith("#") else "# " + l for l in lines]
        out += lines + [""]
    return "\n".join(out).strip("\n") + "\n"


def build_combined(nbdef: dict, profile: str):
    """One Kaggle notebook that runs several former notebooks as sections (asd_lib/asd_runner.py)."""
    slug = nbdef["slug"]
    # Kaggle derives the notebook URL from the title; a different slug makes `kernels status <id>` fail
    title_slug = re.sub(r"[^a-z0-9]+", "-", nbdef["title"].lower()).strip("-")
    assert title_slug == slug, f"title {nbdef['title']!r} gives slug {title_slug!r}, not {slug!r}"
    secs = [{"name": s["name"], "after": s.get("after", []), "rev": s.get("rev", 1)} for s in nbdef["sections"]]
    queue_rev = next((i.get("rev", 1) for i in json.loads((ROOT / "tools" / "queue.json").read_text())["items"]
                      if i["slug"] == slug), 1)
    assert queue_rev == nbdef.get("rev", 1), f"{slug}: rev {nbdef.get('rev', 1)} in combined.json but {queue_rev} in queue.json"
    nb = nbformat.v4.new_notebook()
    rows = "\n".join(f"| `{s['name']}` | `{s['src']}` | {', '.join(s.get('after', [])) or '-'} |" for s in nbdef["sections"])
    nb.cells.append(nbformat.v4.new_markdown_cell(
        f"# {nbdef['title']}\n{nbdef['about']}\n\n"
        "Each section is one former notebook, run as its own process. A section waits for the ones it needs; "
        "a time guard stops before Kaggle's 12-hour limit and the next run continues where this one stopped.\n\n"
        "| section | source | needs |\n|---|---|---|\n" + rows))
    nb.cells.append(nbformat.v4.new_code_cell(BOOTSTRAP.format(profile=profile, user=USER)))
    nb.cells.append(nbformat.v4.new_code_cell(
        "import asd_runner as R\n"
        f"R.setup({USER + '/' + slug!r}, RUN_PROFILE, {json.dumps(secs)}, hours=11.0, rev={nbdef.get('rev', 1)})"))
    datasets = []
    for s in nbdef["sections"]:
        src = SRC / s["src"]
        meta = _read_meta(src)
        vars_ = {}
        if s.get("variant"):
            vars_ = next(v for v in meta["variants"] if v["suffix"] == s["variant"])["vars"]
        title = next((l.lstrip("# ").strip() for kind, lines in split_cells(src.read_text(encoding="utf-8"))
                      if kind == "markdown" for l in lines if l.startswith("# #")), s["src"])
        suffix = f" ({s['variant']})" if s.get("variant") else ""
        nb.cells.append(nbformat.v4.new_markdown_cell(f"## Section `{s['name']}`: {title}{suffix}"))
        nb.cells.append(nbformat.v4.new_code_cell(f"%%writefile {{R.SEC_DIR}}/{s['name']}.py\n" + section_code(src)))
        nb.cells.append(nbformat.v4.new_code_cell(f"R.run({s['name']!r}" + (f", {json.dumps(vars_)}" if vars_ else "") + ")"))
        for ds in meta.get("datasets", []):
            ds = ds.replace("tahidulkabir/", USER + "/")
            if ds not in datasets:
                datasets.append(ds)
    nb.cells.append(nbformat.v4.new_code_cell("R.finish()"))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    d = (OUT if profile == "full" else OUT.parent / f"kaggle_build_{profile}") / slug
    d.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, d / f"{slug}.ipynb")
    gpu = bool(nbdef.get("gpu")) and profile in ("full", "smoke")
    km = {
        "id": f"{USER}/{slug}", "title": nbdef["title"], "code_file": f"{slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True,
        # the orch key lets a run download its own previous output to resume after the time guard
        "dataset_sources": [LIB_DATASET, f"{USER}/asd-xmf-orch-key"] + datasets,
        "kernel_sources": [f"{USER}/{k}" for k in nbdef.get("kernels", [])],
        "competition_sources": [], "model_sources": [],
    }
    if gpu:
        km["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(km, indent=2))
    print(f"built {slug:55s} sections={len(secs)} gpu={gpu} profile={profile}")
    return slug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["full", "preflight", "smoke"])
    ap.add_argument("--only", default=None, help="substring filter on source file names")
    ap.add_argument("--combined", action="store_true", help="build the 4 combined notebooks X1-X4")
    a = ap.parse_args()
    if a.combined:
        for nbdef in json.loads((ROOT / "notebooks" / "combined.json").read_text())["notebooks"]:
            if not a.only or a.only in nbdef["slug"]:
                build_combined(nbdef, a.profile)
        return
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
