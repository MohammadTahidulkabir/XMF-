"""
Runs several former notebooks ("sections") inside one Kaggle notebook (X1-X4).

* every section runs as its own Python process (fresh memory and GPU, a crash never kills the rest)
* a section starts only when the sections it needs finished; dependants of a failed section are skipped
* a time guard stops before Kaggle's 12-hour limit; the next run downloads this notebook's previous
  output and continues (finished sections and finished training jobs are skipped)
* the notebook ends with exactly one marker in /kaggle/working, read by tools/queue_runner.py:
    _xmf_complete_<profile>.txt   all sections finished
    _xmf_resume_<profile>.txt     stopped by the time guard, run it again to continue
    _xmf_failed_<profile>.txt     a section failed (needs a code fix, not a retry)
"""
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import asd_utils as U

T0 = time.time()
MAX_RESUMES = 3
STATE = U.WORK / "_xmf"
SEC_DIR = Path("/tmp/xmf_sections" if U.IS_KAGGLE else tempfile.mkdtemp(prefix="xmf_sections_"))
_cfg = {}


def _marker(kind):
    return U.WORK / f"_xmf_{kind}_{_cfg['profile']}.txt"


def _kaggle(args, timeout=1800):
    try:
        r = subprocess.run(["kaggle"] + args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # CLI missing or timeout
        return 1, repr(e)


def _resume_from_previous(slug):
    """Copy the previous run's output back into /kaggle/working when it stopped on the time guard."""
    key = U.find_path("orch_access_token")
    if key is None:
        print("resume: no token dataset attached - starting from scratch")
        return 0
    os.environ["KAGGLE_API_TOKEN"] = key.read_text().strip()
    if subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"]).returncode != 0:
        return 0
    rc, out = _kaggle(["kernels", "files", slug, "--page-size", "200"], timeout=300)
    name = f"_xmf_resume_{_cfg['profile']}.txt"
    if name not in out:
        print("resume: previous run did not stop on the time guard - starting from scratch")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        rc, out = _kaggle(["kernels", "output", slug, "-p", tmp], timeout=3 * 3600)
        n = 0
        for f in Path(tmp).rglob("*"):
            if f.is_file() and not (f.parent == Path(tmp) and f.suffix == ".log"):  # skip Kaggle's own run log
                t = U.WORK / f.relative_to(tmp)
                t.parent.mkdir(parents=True, exist_ok=True)
                os.replace(f, t)
                n += 1
    count = 0
    old = U.WORK / name
    if old.exists():
        try:
            count = int(old.read_text().split()[0])
        except Exception:
            count = 1
        old.unlink()
    print(f"resume: restored {n} files from the previous run (resume #{count + 1})")
    return count + 1


def setup(slug, profile, sections, hours=11.0):
    """sections: [{"name", "after": [...]}] in run order."""
    _cfg.update(slug=slug, profile=profile, sections=sections, deadline=T0 + hours * 3600, resumes=0)
    STATE.mkdir(parents=True, exist_ok=True)
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    if profile == "full":
        _cfg["resumes"] = _resume_from_previous(slug)
    for kind in ("complete", "resume", "failed"):
        _marker(kind).unlink(missing_ok=True)
    print("sections:", ", ".join(s["name"] for s in sections),
          "| already done:", [s["name"] for s in sections if (STATE / f"{s['name']}.done").exists()])


def _state(name):
    for st in ("done", "failed", "skipped", "stopped"):
        if (STATE / f"{name}.{st}").exists():
            return st
    return "pending"


def _set(name, st, info=""):
    for old in ("failed", "skipped", "stopped"):
        (STATE / f"{name}.{old}").unlink(missing_ok=True)
    (STATE / f"{name}.{st}").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {info}".strip())


def run(name, vars_=None):
    """Run the section script written by the preceding %%writefile cell."""
    sec = next(s for s in _cfg["sections"] if s["name"] == name)
    if _state(name) == "done":
        print(f"[{name}] already done in an earlier run - skipped")
        return
    for old in ("failed", "skipped", "stopped"):
        (STATE / f"{name}.{old}").unlink(missing_ok=True)
    bad = [d for d in sec.get("after", []) if _state(d) != "done"]
    if bad:
        _set(name, "skipped", "needs " + ", ".join(bad))
        print(f"[{name}] skipped: needs {bad}")
        return
    left = _cfg["deadline"] - time.time()
    if left < 300:
        _set(name, "stopped", "no time left")
        print(f"[{name}] not started: time guard")
        return
    prelude = (f"RUN_PROFILE = {_cfg['profile']!r}\nPREFLIGHT = RUN_PROFILE == 'preflight'\n"
               "import os, sys, glob, json, time, warnings\nwarnings.filterwarnings('ignore')\n"
               f"sys.path.insert(0, {str(U.LIB_DIR)!r})\nimport asd_utils as U\n"
               f"VARIANT = {json.dumps(vars_ or {})}\n")
    script = SEC_DIR / f"run_{name}.py"
    script.write_text(prelude + (SEC_DIR / f"{name}.py").read_text(encoding="utf-8"), encoding="utf-8")
    env = {**os.environ, "ASD_SEARCH_WORK": "1", "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
           "TOKENIZERS_PARALLELISM": "false",
           "PYTHONPATH": os.pathsep.join([str(U.LIB_DIR), os.environ.get("PYTHONPATH", "")])}
    logp = STATE / f"{name}.log"
    t0 = time.time()
    print(f"[{name}] start ({left / 3600:.1f} h left in this run)", flush=True)
    with open(logp, "w", encoding="utf-8") as logf:
        p = subprocess.Popen([sys.executable, "-u", str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, env=env, cwd=str(U.WORK), start_new_session=True)
        lines = queue.Queue()
        reader = threading.Thread(target=lambda: [lines.put(l) for l in p.stdout] + [lines.put(None)], daemon=True)
        reader.start()
        timed_out = False
        while True:
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                line = ""
            if line is None:
                p.wait()
                break
            if line:
                print(line, end="", flush=True)
                logf.write(line)
            if time.time() > _cfg["deadline"]:
                timed_out = True
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                    time.sleep(20)
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:
                    p.kill()
                p.wait()
                break
    mins = (time.time() - t0) / 60
    if timed_out:
        _set(name, "stopped", f"time guard after {mins:.0f} min")
        print(f"[{name}] STOPPED by the time guard after {mins:.0f} min - the next run continues it")
    elif p.returncode == 0:
        _set(name, "done", f"{mins:.1f} min")
        print(f"[{name}] done in {mins:.1f} min")
    else:
        _set(name, "failed", f"exit {p.returncode} after {mins:.1f} min")
        print(f"[{name}] FAILED (exit {p.returncode}) - log: {logp}")
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def finish():
    st = {s["name"]: _state(s["name"]) for s in _cfg["sections"]}
    U.save_json({"profile": _cfg["profile"], "sections": st, "resumes": _cfg["resumes"],
                 "hours": round((time.time() - T0) / 3600, 2)}, U.WORK / "_xmf_status.json")
    print(json.dumps(st, indent=1))
    failed = [k for k, v in st.items() if v in ("failed", "skipped")]
    stopped = [k for k, v in st.items() if v in ("stopped", "pending")]
    if failed:
        _marker("failed").write_text("failed: " + ", ".join(failed))
        print("RESULT: FAILED sections", failed, "- see _xmf/<section>.log")
    elif stopped:
        if _cfg["resumes"] >= MAX_RESUMES:
            _marker("failed").write_text(f"still unfinished after {MAX_RESUMES} resumes: " + ", ".join(stopped))
            print("RESULT: FAILED - too many resumes")
        else:
            _marker("resume").write_text(f"{_cfg['resumes']} resume(s) so far; left: " + ", ".join(stopped))
            print("RESULT: stopped by the time guard; the queue runs this notebook again to continue", stopped)
    else:
        _marker("complete").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        print("RESULT: COMPLETE")
