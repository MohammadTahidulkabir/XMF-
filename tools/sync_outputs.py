"""
Copy the outputs of every finished queue notebook from Kaggle into outputs/<slug>/ of the repo.
Small result files only (tables, figures, json, logs); weights and big arrays stay on Kaggle.
A notebook is synced again only when Kaggle has a newer finished run.
"""
import json
import shutil
import tempfile
from pathlib import Path

import kaggle_ops as K
import queue_runner as Q

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
KEEP = {".csv", ".json", ".md", ".png", ".txt", ".log", ".html"}
MAX_MB = 20


def main():
    items = json.loads(Q.QUEUE.read_text())["items"]
    st = Q.snapshot(items)
    OUT.mkdir(exist_ok=True)
    import os
    tok = os.environ.get("KAGGLE_API_TOKEN", "")
    rc, cfg = K.sh(["config", "view"])
    rc2, raw = K.sh(["kernels", "status", f"{K.USER}/{items[0]['slug']}"])
    (OUT / "_kaggle_check.txt").write_text(
        f"secret present: {bool(tok)} | starts with KGAT_: {tok.startswith('KGAT_')} | length: {len(tok)}
"
        + "
".join(l for l in cfg.splitlines() if "username" in l) + "
" + raw[-400:] + "
")
    (OUT / "_queue_status.json").write_text(json.dumps(st, indent=1))
    for i in items:
        slug = i["slug"]
        if st[slug] not in ("DONE", "FAILED"):
            continue
        dst = OUT / slug
        if (dst / "_synced.txt").exists():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            K.sh(["kernels", "output", f"{K.USER}/{slug}", "-p", tmp], timeout=3600)
            for f in Path(tmp).rglob("*"):
                if f.is_file() and f.suffix.lower() in KEEP and f.stat().st_size < MAX_MB * 1e6:
                    t = dst / f.relative_to(tmp)
                    t.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, t)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "_synced.txt").write_text(st[slug])
        print("synced", slug, st[slug])


if __name__ == "__main__":
    main()
