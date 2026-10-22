"""
Push notebooks to Kaggle.

  python tools/push.py lib                         # new version of the asd-xmf-lib dataset
  python tools/push.py preflight <slug> [...]      # CPU smoke run (no GPU quota), waits, prints log tail on error
  python tools/push.py full <slug> [...]           # GPU run (normally done by queue_runner.py)
"""
import sys
from pathlib import Path

import kaggle_ops as K


def main():
    mode, slugs = sys.argv[1], sys.argv[2:]
    if mode == "lib":
        rc, out = K.sh(["datasets", "version", "-p", str(K.ROOT / "asd_lib"), "-m", " ".join(slugs) or "update", "-q"], timeout=900)
        print(out)
        return
    for slug in slugs:
        print(K.push(slug, mode))
        st = K.wait(f"{K.USER}/{slug}", poll=30)
        print(slug, "->", st)
        if st != "COMPLETE":
            log = K.fetch_log(slug, K.ROOT / "logs" / slug)
            print("----- log tail -----\n" + log[-4000:])
            sys.exit(1)


if __name__ == "__main__":
    main()
