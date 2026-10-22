"""
ASD-XMF GPU queue - stateless, so it can run on your PC *or* inside Kaggle.

Progress is read from Kaggle itself, nothing is kept locally:
  * every notebook writes `_run_profile_full.txt` at the start of a full run
  * item DONE    = latest run COMPLETE and that marker is in its output
  * item FAILED  = latest run ERROR/CANCELLED and the marker is present (blocks dependants, no retry loop)
  * otherwise PENDING (only preflight runs so far, or never run)

Rules (start only after the harness work is done, one at a time):
  * busy = any kernel on the account that is not part of this queue is RUNNING/QUEUED
  * push the next ready item only after QUIET_CHECKS consecutive quiet checks (POLL apart)
  * never push while one of our own items is RUNNING/QUEUED
  * HOLD file next to this script pauses; FORCE ignores foreign kernels

Usage
  python tools/queue_runner.py --loop                    # on the PC
  (cloud) the asd-xmf-queue-orchestrator notebook runs:  --loop --max-hours 11 --self-slug asd-xmf-queue-orchestrator
"""
import argparse
import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import kaggle_ops as K

TOOLS = Path(__file__).resolve().parent
QUEUE = Path(os.environ.get("ASD_QUEUE_FILE", TOOLS / "queue.json"))
LOG = Path(os.environ.get("ASD_QUEUE_LOG", TOOLS / "queue_log.txt"))
POLL = int(os.environ.get("ASD_QUEUE_POLL", 300))
QUIET_CHECKS = 3
MARKER = "_run_profile_full.txt"
ORCH = "asd-xmf-queue-orchestrator"
FAILED = {"ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED", "CANCEL_REQUESTED"}


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


LEDGER = Path(os.environ.get("ASD_QUEUE_LEDGER", TOOLS / "done_ledger.json"))


def _ledger():
    try:
        return json.loads(LEDGER.read_text())
    except Exception:
        return {}


def _ledger_add(slug, state):
    d = _ledger()
    d[slug] = {"state": state, "t": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        LEDGER.write_text(json.dumps(d, indent=1))
    except OSError:
        pass


def has_full_marker(slug: str):
    """True / False, or None when Kaggle could not be asked (network down, timeout)."""
    with tempfile.TemporaryDirectory() as d:
        rc, out = K.sh(["kernels", "output", f"{K.USER}/{slug}", "-p", d, "--file-pattern", MARKER.replace(".", r"\.")], timeout=600)
        if any(p.name == MARKER for p in Path(d).rglob("*")):
            return True
        return False if rc == 0 else None


def item_state(slug: str) -> str:
    led = _ledger().get(slug, {}).get("state")
    s = K.status(f"{K.USER}/{slug}")
    if s in K.ACTIVE:
        return "RUNNING"
    if led in ("DONE", "FAILED"):  # finished full runs are never repeated
        return led
    if s.startswith("UNKNOWN"):
        return "UNKNOWN"
    if s == "COMPLETE" or s in FAILED:
        m = has_full_marker(slug)
        if m is None:
            return "UNKNOWN"
        if m:
            st = "DONE" if s == "COMPLETE" else "FAILED"
            _ledger_add(slug, st)
            return st
    return "PENDING"


def foreign_active(ours):
    act = []
    for r in K.recent_kernels(25):
        ref = r.get("ref", "")
        if not ref or ref.split("/")[-1] in ours:
            continue
        s = K.status(ref)
        if s in K.ACTIVE:
            act.append(f"{ref}:{s}")
    return act


def snapshot(items):
    st = {i["slug"]: item_state(i["slug"]) for i in items}
    for _ in range(len(items)):  # propagate failures to dependants
        for i in items:
            if st[i["slug"]] == "PENDING" and any(st.get(d) in ("FAILED", "BLOCKED") for d in i.get("after", [])):
                st[i["slug"]] = "BLOCKED"
    return st


def loop(max_hours=None, self_slug=None):
    t0, quiet, last_summary = time.time(), 0, None
    log(f"queue runner started (max_hours={max_hours}, self={self_slug})")
    while True:
        try:
            if max_hours and time.time() - t0 > max_hours * 3600:
                if self_slug:
                    log("time limit reached - relaunching the orchestrator as a new Kaggle run")
                    log(K.push_dir(Path(os.environ["ASD_SELF_DIR"]))[-300:])
                return
            items = json.loads(QUEUE.read_text())["items"]
            ours = {i["slug"] for i in items} | {ORCH}
            if not self_slug and (cloud_runner_active() or K.status(f"{K.USER}/{ORCH}") in K.ACTIVE):
                if quiet >= 0:
                    log("cloud orchestrator is running on Kaggle - PC runner standing by")
                quiet = -1  # stand-by: the cloud orchestrator owns the queue
                time.sleep(POLL)
                continue
            quiet = max(quiet, 0)
            if (TOOLS / "HOLD").exists():
                time.sleep(POLL)
                continue
            st = snapshot(items)
            summary = {k: sum(v == k for v in st.values()) for k in ("DONE", "RUNNING", "PENDING", "FAILED", "BLOCKED", "UNKNOWN")}
            if summary != last_summary:
                log(f"status {summary} | " + ", ".join(f"{s.replace('asd-xmf-', '')}={v}" for s, v in st.items() if v != "PENDING"))
                last_summary = summary
            if summary["RUNNING"]:
                time.sleep(POLL)
                continue
            ready = [i for i in items if st[i["slug"]] == "PENDING"
                     and all(st.get(d) == "DONE" for d in i.get("after", []))]
            if not ready:
                if not summary["PENDING"]:
                    log("queue finished - nothing left to run")
                    return
                time.sleep(POLL)
                continue
            busy = [] if (TOOLS / "FORCE").exists() else foreign_active(ours)
            if busy:
                if quiet:
                    log(f"harness/other kernels active, waiting: {busy}")
                quiet = 0
                time.sleep(POLL)
                continue
            quiet += 1
            if quiet < (1 if (TOOLS / "FORCE").exists() else QUIET_CHECKS):
                time.sleep(POLL)
                continue
            nxt = ready[0]["slug"]
            out = K.push(nxt, "full")
            log(f"PUSH {nxt}: {out[-250:]}")
            if "quota" in out.lower() or "429" in out:
                log("GPU quota / rate limit - sleeping 3 h")
                time.sleep(3 * 3600)
            quiet = 0
            time.sleep(POLL)
        except Exception:
            log("runner error:\n" + traceback.format_exc()[-1500:])
            time.sleep(POLL)


def tick() -> str:
    """One stateless step (used by GitHub Actions every 20 min): push the next ready
    notebook if none of ours is running.  The project account is dedicated, so other
    kernels are not waited for when ASD_FORCE=1."""
    if (TOOLS / "HOLD").exists():
        log("tick: HOLD file present - nothing pushed")
        return "hold"
    items = json.loads(QUEUE.read_text())["items"]
    st = snapshot(items)
    summary = {k: sum(v == k for v in st.values()) for k in ("DONE", "RUNNING", "PENDING", "FAILED", "BLOCKED", "UNKNOWN")}
    log(f"tick status {summary}")
    if summary["RUNNING"]:
        return "wait"
    ready = [i for i in items if st[i["slug"]] == "PENDING" and all(st.get(d) == "DONE" for d in i.get("after", []))]
    if not ready:
        return "finished" if not summary["PENDING"] else "blocked"
    if os.environ.get("ASD_FORCE") != "1" and foreign_active({i["slug"] for i in items} | {ORCH}):
        return "busy"
    out = K.push(ready[0]["slug"], "full")
    log(f"PUSH {ready[0]['slug']}: {out[-200:]}")
    return "pushed"


def cloud_runner_active(repo="MohammadTahidulkabir/XMF-", minutes=45) -> bool:
    """True when the GitHub Actions queue ran recently -> the PC runner stands by."""
    try:
        import urllib.request, datetime
        with urllib.request.urlopen(f"https://api.github.com/repos/{repo}/actions/runs?per_page=1", timeout=20) as r:
            runs = json.loads(r.read()).get("workflow_runs", [])
        if not runs:
            return False
        t = datetime.datetime.strptime(runs[0]["created_at"], "%Y-%m-%dT%H:%M:%SZ")
        if (datetime.datetime.utcnow() - t).total_seconds() > minutes * 60:
            return False
        # only defer when the cloud run can actually talk to Kaggle (not all UNKNOWN)
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/main/outputs/_queue_status.json", timeout=20) as r:
            st = json.loads(r.read())
        return any(v != "UNKNOWN" for v in st.values())
    except Exception:
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--status", action="store_true", help="print item states once and exit")
    ap.add_argument("--tick", action="store_true", help="one step (GitHub Actions)")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--self-slug", default=None)
    a = ap.parse_args()
    if a.tick:
        print(tick())
    elif a.status:
        print(json.dumps(snapshot(json.loads(QUEUE.read_text())["items"]), indent=1))
    else:
        loop(a.max_hours, a.self_slug)
