# %% [meta]
# {"slug": "asd-xmf-queue-orchestrator", "title": "ASD-XMF Queue Orchestrator",
#  "gpu": false, "internet": true,
#  "datasets": ["tahidulkabir/asd-xmf-queue-bundle"], "kernels": [], "phase": "Q"}

# %% [markdown]
# # ASD-XMF queue orchestrator (runs on Kaggle CPU, no GPU quota)
# Watches the account and pushes the ASD-XMF GPU notebooks one by one, **only after the
# harness / any other notebook has finished** and the account has been quiet for 15 minutes.
# Before Kaggle's 12-hour limit it relaunches itself, so the queue keeps going while your PC
# and Claude are off.  It stops by itself when every notebook is done.
#
# One-time setup: *Add-ons -> Secrets* -> add `KAGGLE_API_TOKEN` (contents of your
# `~/.kaggle/access_token`) and tick it for this notebook.
# To stop the queue: stop this notebook's run on its Kaggle page.

# %%
import os, sys, subprocess, shutil
from pathlib import Path
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
except Exception as e:
    TOKEN = None
    print("Secret KAGGLE_API_TOKEN is not attached to this notebook:", repr(e)[:200])
    print("Add it under Add-ons -> Secrets, tick it for this notebook, then Save & Run All once.")

# %%
if TOKEN:
    os.environ["KAGGLE_API_TOKEN"] = TOKEN.strip()
    kd = Path.home() / ".kaggle"; kd.mkdir(exist_ok=True)
    (kd / "access_token").write_text(TOKEN.strip()); os.chmod(kd / "access_token", 0o600)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=False)
    bundle = U.find_path("queue.json").parent.parent          # <bundle>/tools/queue.json
    os.environ.update({"ASD_QUEUE_ROOT": str(bundle), "ASD_QUEUE_FILE": str(bundle / "tools" / "queue.json"),
                       "ASD_QUEUE_LOG": "/kaggle/working/queue_log.txt",
                       "ASD_SELF_DIR": str(bundle / "kaggle_build" / "asd-xmf-queue-orchestrator")})
    sys.path.insert(0, str(bundle / "tools"))
    import queue_runner as Q
    print("kaggle CLI:", subprocess.run(["kaggle", "--version"], capture_output=True, text=True).stdout.strip())
    Q.loop(max_hours=0.2 if PREFLIGHT else 11.0, self_slug="asd-xmf-queue-orchestrator")
