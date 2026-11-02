"""
Runs several former notebooks ("sections") inside one Kaggle notebook (X1-X4).

* every section runs as its own Python process (fresh memory and GPU, a crash never kills the rest)
* a section starts only when the sections it needs finished; dependants of a failed section are skipped
* a time guard stops before Kaggle's 12-hour limit; the next run downloads this notebook's previous
  output and continues (finished sections and finished training jobs are skipped)
* revisions: every section has a `rev` (notebooks/combined.json).  A new run reuses the previous run's finished
  sections whose rev is unchanged (and whose inputs were reused); a section with a new rev runs again
* the notebook ends with exactly one marker in /kaggle/working, read by tools/queue_runner.py
  (<r> is "" for notebook rev 1, else "_r<rev>"):
    _xmf_complete_<profile><r>.txt   all sections finished
    _xmf_resume_<profile><r>.txt     stopped by the time guard, run it again to continue
    _xmf_failed_<profile><r>.txt     a section failed (needs a code fix, not a retry)
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
    r = "" if _cfg["rev"] == 1 else f"_r{_cfg['rev']}"
    return U.WORK / f"_xmf_{kind}_{_cfg['profile']}{r}.txt"


def _kaggle(args, timeout=1800):
    try:
        r = subprocess.run(["kaggle"] + args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # CLI missing or timeout
        return 1, repr(e)


def _resume_from_previous(slug):
    """Copy the previous full run's output back into /kaggle/working (time guard stop, or an earlier revision whose
    unchanged sections can be reused).  Returns how many time-guard resumes this revision has had."""
    key = U.find_path("orch_access_token")
    if key is None:
        print("resume: no token dataset attached - starting from scratch")
        return 0
    os.environ["KAGGLE_API_TOKEN"] = key.read_text().strip()
    if subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"]).returncode != 0:
        return 0
    rc, out = _kaggle(["kernels", "files", slug, "--page-size", "200"], timeout=300)
    prof = _cfg["profile"]
    if not any(f"_xmf_{k}_{prof}" in out for k in ("complete", "resume", "failed")):
        print("resume: no earlier full run of this notebook - starting from scratch")
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
    same = _marker("resume")
    if same.exists():  # stopped by the time guard in this same revision
        try:
            count = int(same.read_text().split()[0]) + 1
        except Exception:
            count = 1
    for old in U.WORK.glob(f"_xmf_*_{prof}*.txt"):
        old.unlink()
    print(f"resume: restored {n} files from the previous run" + (f" (time-guard resume #{count})" if count else ""))
    return count


def _done_rev(name):
    try:
        txt = (STATE / f"{name}.done").read_text()
        return int(txt.split("rev=")[1].split()[0]) if "rev=" in txt else 1
    except Exception:
        return None


def setup(slug, profile, sections, hours=11.0, rev=1):
    """sections: [{"name", "after": [...], "rev": n}] in run order; rev = notebook revision (queue markers)."""
    _cfg.update(slug=slug, profile=profile, sections=sections, deadline=T0 + hours * 3600, resumes=0, rev=rev)
    STATE.mkdir(parents=True, exist_ok=True)
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    if profile == "full":
        _cfg["resumes"] = _resume_from_previous(slug)
    redo = set()
    for sec in sections:  # in run order, so a re-run section also re-runs everything that needs it
        name = sec["name"]
        if not (STATE / f"{name}.done").exists():
            continue
        if _done_rev(name) != sec.get("rev", 1) or any(d in redo for d in sec.get("after", [])):
            (STATE / f"{name}.done").unlink()
            redo.add(name)
    for kind in ("complete", "resume", "failed"):
        _marker(kind).unlink(missing_ok=True)
    print("sections:", ", ".join(f"{s['name']}(rev {s.get('rev', 1)})" for s in sections),
          "| reused from earlier runs:", [s["name"] for s in sections if (STATE / f"{s['name']}.done").exists()],
          "| re-run (new revision):", sorted(redo))


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
        # not run as __main__: "spawn" worker pools (NB01, D1) re-import a main *file* and would re-run the whole
        # section; with `-c` there is no main file, exactly like a notebook kernel
        launch = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__xmf_section__')"
        p = subprocess.Popen([sys.executable, "-u", "-c", launch, str(script)], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, env=env, cwd=str(U.WORK), start_new_session=True)
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
        _set(name, "done", f"{mins:.1f} min rev={sec.get('rev', 1)}")
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
