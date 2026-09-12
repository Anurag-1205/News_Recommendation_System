"""Refresh the inline copy of src/baselines/ebrec_compat.py inside nrms_smoke.py.
Run after editing the shim:  .venv/bin/python scripts/kaggle/nrms_smoke/inline_shim.py
Exists because a Kaggle script kernel uploads one file, and the shim is not yet fetchable
from GitHub. The copy is base64 so the shim's own quotes cannot break the kernel file. Once the
branch is pushed, the kernel can fetch the raw file instead."""
import base64, pathlib, re

root = pathlib.Path(__file__).resolve().parents[3]
shim = (root / "src/baselines/ebrec_compat.py").read_bytes()
b64 = base64.b64encode(shim).decode()
k = root / "scripts/kaggle/nrms_smoke/nrms_smoke.py"
s = k.read_text()
new, n = re.subn(r'EBREC_COMPAT_B64 = "[A-Za-z0-9+/=]*"', f'EBREC_COMPAT_B64 = "{b64}"', s)
assert n == 1, n
k.write_text(new)
print("inline shim refreshed,", len(shim), "bytes")
