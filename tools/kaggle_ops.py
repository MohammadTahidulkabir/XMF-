"""
Thin wrappers around the Kaggle CLI used by push.py and queue_runner.py.
"""
import csv
import io
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import os
import tempfile
ROOT = Path(os.environ.get("ASD_QUEUE_ROOT", Path(__file__).resolve().parents[1]))
# ---- project account (kept separate from ~/.kaggle) -------------------------
ACCT_DIR = Path(os.environ.get("ASD_KAGGLE_DIR", "D:/kaggle_project"))
def _acct_user(default):
    try:
        return json.loads((ACCT_DIR / "account.json").read_text())["user"]
    except Exception:
        return default
USER = os.environ.get("ASD_KAGGLE_USER") or _acct_user("tahidulkabir")
KAGGLE = shutil.which("kaggle") or "kaggle"
DONE = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED", "CANCEL_REQUESTED"}
ACTIVE = {"RUNNING", "QUEUED", "NEW", "PENDING"}


def sh(args, timeout=600):
    try:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}  # Windows charmap crashes otherwise
        if (ACCT_DIR / "access_token").exists():
            # this project's own Kaggle account; pass the token explicitly so the CLI can
            # never fall back to the login cached for your main account
            env["KAGGLE_CONFIG_DIR"] = str(ACCT_DIR)
            env["KAGGLE_API_TOKEN"] = (ACCT_DIR / "access_token").read_text().strip()
        r = subprocess.run([KAGGLE] + args, capture_output=True, text=True, timeout=timeout, encoding="utf-8",
                           errors="replace", env=env)
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT"
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _sh_unused(args, timeout=600):
    r = subprocess.run([KAGGLE] + args, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def status(ref: str) -> str:
    rc, out = sh(["kernels", "status", ref], timeout=120)
    m = re.search(r'status "(?:KernelWorkerStatus\.)?(\w+)"', out)
    if m:
        return m.group(1).upper()
    low = out.lower()
    if "404" in out or "not found" in low or "wrong kernel slug" in low or "permission denied" in low:
        return "MISSING"
    return "UNKNOWN:" + out.strip()[-120:]


def push_dir(d: Path) -> str:
    """Push from a writable temp copy (inputs are read-only inside Kaggle)."""
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / Path(d).name
        shutil.copytree(d, dst)
        rc, out = sh(["kernels", "push", "-p", str(dst)], timeout=900)
    return out.strip()


def push(slug: str, profile: str = "full") -> str:
    return push_dir(ROOT / ("kaggle_build" if profile == "full" else f"kaggle_build_{profile}") / slug)


def recent_kernels(n: int = 25):
    rc, out = sh(["kernels", "list", "--mine", "--sort-by", "dateRun", "--page-size", str(n), "--csv"], timeout=180)
    rows = list(csv.DictReader(io.StringIO(out[out.find("ref"):]))) if "ref" in out else []
    return rows


def fetch_log(slug: str, dest: Path) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    sh(["kernels", "output", f"{USER}/{slug}", "-p", str(dest), "--file-pattern", r".*\.log$"], timeout=900)
    logs = sorted(dest.glob("*.log"))
    if not logs:
        return ""
    txt = logs[0].read_text(encoding="utf-8", errors="replace")
    try:  # Kaggle logs are a JSON list of {stream_name, time, data}
        items = json.loads(txt)
        txt = "".join(i.get("data", "") for i in items)
    except Exception:
        pass
    return txt


def wait(ref: str, poll: int = 60, max_hours: float = 13) -> str:
    t0 = time.time()
    last = None
    while time.time() - t0 < max_hours * 3600:
        s = status(ref)
        if s != last:
            print(time.strftime("%H:%M:%S"), ref, s, flush=True)
            last = s
        if s in DONE:
            return s
        time.sleep(poll)
    return "TIMEOUT"
