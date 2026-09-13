#!/usr/bin/env python3
"""Embed local files into a kernel so it can run code that is committed but not yet pushed.

    .venv/bin/python scripts/kaggle/overlay.py scripts/kaggle/nrms_ebnerd_fresh/nrms_ebnerd_fresh.py \\
        src/baselines/nrms_fresh_features.py tests/test_nrms_fresh_model_mind.py
    .venv/bin/python scripts/kaggle/overlay.py scripts/kaggle/nrms_ebnerd_fresh/nrms_ebnerd_fresh.py --clear

Writes/replaces a block `OVERLAY = {...}` (path → base64 of the file at HEAD) plus the HEAD
commit into the kernel. The kernel writes those files over its fresh clone right after
cloning, prints each path with its sha256 and `OVERLAY_FROM <commit>`, and puts both into the
manifests. Used only while the agent cannot push (CONTEXT.md C-027 note); `--clear` removes it.
"""
import base64, hashlib, re, subprocess, sys
from pathlib import Path

kernel = Path(sys.argv[1]); files = sys.argv[2:]
s = kernel.read_text()
s = re.sub(r"\n# ---- overlay \(unpushed commit\) .*?# ---- end overlay ----\n", "\n", s, flags=re.S)
if files != ["--clear"]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    entries = ",\n".join(f'    "{f}": "{base64.b64encode(Path(f).read_bytes()).decode()}"' for f in files)
    block = f'''
# ---- overlay (unpushed commit) ------------------------------------------------------------------
OVERLAY_FROM = "{head}"
OVERLAY = {{
{entries}
}}
# ---- end overlay ----
'''
    s = s.replace("\nT0 = time.time()\n", block + "\nT0 = time.time()\n", 1)
kernel.write_text(s)
print("overlay", "cleared" if files == ["--clear"] else f"{len(files)} file(s) from {head[:7]}")
