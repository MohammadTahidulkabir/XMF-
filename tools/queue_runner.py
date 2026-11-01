"""
ASD-XMF GPU queue - stateless, so it can run on your PC *or* inside Kaggle.

Progress is read from Kaggle itself, nothing is kept locally:
  * single notebooks (D1) write `_run_profile_full.txt` at the start of a full run:
    DONE = latest run COMPLETE with that marker, FAILED = ERROR/CANCELLED with it
  * combined notebooks (X1-X4, asd_lib/asd_runner.py) end with one marker:
    `_xmf_complete_full.txt` DONE, `_xmf_failed_full.txt` FAILED, `_xmf_resume_full.txt` run again to continue
  * otherwise PENDING (only preflight runs so far, or never run)
  * right after a push Kaggle can still report the previous run, so a pushed item counts as RUNNING until its
    new run has been seen active (or PUSH_GRACE has passed); at most MAX_PUSHES pushes per item and runner

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
import re
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
RETRIES = {}   # pushes made by THIS runner instance per notebook (protects against killed-run loops)
PUSHED = {}    # slug -> time of our last push
SEEN_ACTIVE = set()  # slugs whose run after our last push has been seen RUNNING/QUEUED
PUSH_GRACE = 45 * 60
MAX_PUSHES = 4
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


def has_file(slug: str, names):
    """The first of `names` in the latest run's output, "" if none, None when Kaggle could not be asked.
    Uses the output *file listing*: `kernels output --file-pattern` silently matches nothing in Kaggle CLI 2.2.x."""
    names = [names] if isinstance(names, str) else list(names)
    token, found = None, set()
    for _ in range(50):
        args = ["kernels", "files", f"{K.USER}/{slug}", "--page-size", "200"] + (["--page-token", token] if token else [])
        rc, out = K.sh(args, timeout=300)
        if rc != 0 and "name" not in out:
            return None
        found |= {n for n in names if re.search(r"(^|/|\s)" + re.escape(n) + r"\s", out)}
        m = re.search(r"Next Page Token = (\S+)", out)
        if not m:
            break
        token = m.group(1)
    return next((n for n in names if n in found), "")


def has_full_marker(slug: str):
    m = has_file(slug, MARKER)
    return None if m is None else bool(m)


def combined_state(slug: str) -> str:
    """State of an X1-X4 notebook whose latest run is not active."""
    m = has_file(slug, ["_xmf_complete_full.txt", "_xmf_failed_full.txt", "_xmf_resume_full.txt", MARKER])
    if m is None:
        return "UNKNOWN"
    if m == "_xmf_complete_full.txt":
        _ledger_add(slug, "DONE")
        return "DONE"
    if m == "_xmf_failed_full.txt":
        _ledger_add(slug, "FAILED")
        return "FAILED"
    if m == MARKER and RETRIES.get(slug, 0) >= 2:
        # a full run that ended without a result marker was killed (memory, 12 h limit) - twice now
        _ledger_add(slug, "FAILED")
        return "FAILED"
    return "PENDING"  # never run, preflight only, or stopped by its time guard (the next run continues)


def item_state(slug: str, kind: str = "single") -> str:
    led = _ledger().get(slug, {}).get("state")
    s = K.status(f"{K.USER}/{slug}")
    if s in K.ACTIVE:
        if slug in PUSHED:
            SEEN_ACTIVE.add(slug)
        return "RUNNING"
    if led in ("DONE", "FAILED"):  # finished full runs are never repeated
        return led
    if slug in PUSHED and slug not in SEEN_ACTIVE and time.time() - PUSHED[slug] < PUSH_GRACE:
        return "RUNNING"  # just pushed: Kaggle may still report the previous run
    if s.startswith("UNKNOWN"):
        return "UNKNOWN"
    if s == "MISSING" and slug not in PUSHED:
        return "PENDING"  # never run under this slug (its file listing fails too)
    st = combined_state(slug) if kind == "combined" else None
    if st in ("DONE", "FAILED", "UNKNOWN"):
        return st
    if RETRIES.get(slug, 0) >= MAX_PUSHES:
        _ledger_add(slug, "FAILED")
        log(f"{slug}: pushed {MAX_PUSHES} times without a result - marked FAILED, check it on Kaggle")
        return "FAILED"
    if st:
        return st
    if s == "COMPLETE" or s in FAILED:
        m = has_full_marker(slug)
        if m is None:
            return "UNKNOWN"
        if m:
            st = "DONE" if s == "COMPLETE" else "FAILED"
            _ledger_add(slug, st)
            return st
        if s in FAILED and RETRIES.get(slug, 0) >= 1:
            # a run we started was killed (out of memory / time) and left no output: retry once, then stop
            _ledger_add(slug, "FAILED")
            return "FAILED"
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
    st = {i["slug"]: item_state(i["slug"], i.get("kind", "single")) for i in items}
    for i in items:
        # an active run we did not push whose inputs are not finished can only be a preflight test:
        # it must not hold back the queue
        if st[i["slug"]] == "RUNNING" and i["slug"] not in PUSHED and \
                any(st.get(d) != "DONE" for d in i.get("after", [])):
            st[i["slug"]] = "PENDING"
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
            out = push_item(ready[0]["slug"])
            if rate_limited(out):
                left = max_hours * 3600 - (time.time() - t0) - 600 if max_hours else 3 * 3600
                nap = max(POLL, min(3 * 3600, left))  # never sleep past the orchestrator's own relaunch
                log(f"GPU quota / rate limit - sleeping {nap / 3600:.1f} h")
                time.sleep(nap)
            quiet = 0
            time.sleep(POLL)
        except Exception:
            log("runner error:\n" + traceback.format_exc()[-1500:])
            time.sleep(POLL)


def push_item(slug: str) -> str:
    RETRIES[slug] = RETRIES.get(slug, 0) + 1
    out = K.push(slug, "full")
    if "successfully pushed" in out:
        PUSHED[slug] = time.time()
        SEEN_ACTIVE.discard(slug)
    log(f"PUSH {slug} (#{RETRIES[slug]}): {out[-250:]}")
    return out


def rate_limited(out: str) -> bool:
    """Only a refused push counts ("Kernel version 2429 successfully pushed" once matched a bare "429")."""
    if "successfully pushed" in out:
        return False
    low = out.lower()
    return "quota" in low or "too many requests" in low or "rate limit" in low or re.search(r"\b429\b", out) is not None


def tick() -> str:
    """One step for GitHub Actions (every 20 min), a watchdog only.  The orchestrator notebook on Kaggle is the
    single process that pushes queue notebooks (two pushers restarted each other's runs); this step relaunches it
    when it is not running and work is left."""
    if (TOOLS / "HOLD").exists():
        log("tick: HOLD file present - nothing pushed")
        return "hold"
    items = json.loads(QUEUE.read_text())["items"]
    st = snapshot(items)
    summary = {k: sum(v == k for v in st.values()) for k in ("DONE", "RUNNING", "PENDING", "FAILED", "BLOCKED", "UNKNOWN")}
    log(f"tick status {summary}")
    if not (summary["PENDING"] or summary["RUNNING"] or summary["UNKNOWN"]):
        return "finished"
    o = K.status(f"{K.USER}/{ORCH}")
    if o in K.ACTIVE or o.startswith("UNKNOWN"):
        return f"orchestrator {o}"
    out = K.push(ORCH, "full")
    log(f"tick: orchestrator was {o} - relaunched: {out[-200:]}")
    return "relaunched orchestrator"


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
