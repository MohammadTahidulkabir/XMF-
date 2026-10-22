"""
Execute the preflight notebooks locally (CPU) with a Kaggle-like input
layout, to catch bugs before spending any Kaggle time.

    python tools/build.py --profile preflight
    python tools/local_run.py asd-xmf-nb13-gaze-eyetracking asd-xmf-nb00-data-audit-manifest ...

Datasets map to local folders (LOCAL_DATA); kernel sources map to the work
folder of an earlier local run.  Directory junctions are used (no admin).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "kaggle_build_preflight"
RUN = Path(os.environ.get("ASD_LOCAL_RUN", "D:/asd_test/run"))
DATA = Path(os.environ.get("ASD_LOCAL_DATA", "D:/asd_test/data"))
LOCAL_DATA = {
    "tahidulkabir/asd-xmf-lib": ROOT / "asd_lib",
    "tahidulkabir/asd-xmf-recanvo": Path("D:/asd_xmf_data/recanvo_kaggle"),
    "shuvoalok/raf-db-dataset": DATA / "raf-db-dataset",
    "msambare/fer2013": DATA / "fer2013",
    "imtkaggleteam/eye-tracking-autism": DATA / "eye-tracking-autism",
    "warcoder/infant-cry-audio-corpus": DATA / "infant-cry-audio-corpus",
}


def junction(link: Path, target: Path):
    if link.exists():
        return
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


def run(slug: str, timeout: int = 5400) -> bool:
    meta = json.loads((BUILD / slug / "kernel-metadata.json").read_text())
    base = RUN / slug
    inp, work = base / "input", base / "work"
    inp.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    for ds in meta["dataset_sources"]:
        tgt = LOCAL_DATA.get(ds)
        if tgt is None or not tgt.exists():
            print(f"  [warn] no local copy of dataset {ds}")
            continue
        junction(inp / ds.split("/")[1], tgt)
    for ks in meta["kernel_sources"]:
        dep = RUN / ks.split("/")[1] / "work"
        if not dep.exists():
            print(f"  [warn] dependency {ks} has not been run locally")
            continue
        junction(inp / ks.split("/")[1], dep)
    env = {"ASD_INPUT": str(inp), "ASD_WORK": str(work), "HF_HOME": "D:/hf_cache", "TORCH_HOME": "D:/torch_cache",
           "PIP_CACHE_DIR": "D:/pip_cache", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    os.environ.update(env)
    nb = nbformat.read(BUILD / slug / f"{slug}.ipynb", 4)
    t0 = time.time()
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(BUILD / slug)}})
    ok = True
    try:
        client.execute()
    except Exception as e:
        ok = False
        print(f"  [FAIL] {slug}: {str(e)[-3000:]}")
    nbformat.write(nb, base / f"{slug}.executed.ipynb")
    print(f"{'OK ' if ok else 'ERR'} {slug}  {time.time()-t0:.0f}s  -> {base}")
    return ok


if __name__ == "__main__":
    results = {s: run(s) for s in sys.argv[1:]}
    print(json.dumps(results, indent=1))
    sys.exit(0 if all(results.values()) else 1)
