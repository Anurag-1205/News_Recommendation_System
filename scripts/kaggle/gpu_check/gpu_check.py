"""P0 step 3: prove that a Kaggle kernel on this account sees 2x T4 and runs an fp16 matmul.

Pushed with:  .venv/bin/kaggle kernels push -p scripts/kaggle/gpu_check --accelerator NvidiaTeslaT4
Fetched with: .venv/bin/kaggle kernels output <user>/a2-gpu-check -p <dir>

Prints one line per check so the log is the evidence (RESULTS.md P0). Exits non-zero on any
failure, so `kaggle kernels status` reports "error" rather than a green run with a bad number.
"""
import subprocess
import sys
import time

import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())

n = torch.cuda.device_count()
print("device_count", n)
ok = n == 2 and all("T4" in torch.cuda.get_device_name(i) for i in range(n))

# fp16 autocast + GradScaler is the training recipe (CLAUDE.md §6). Time a 4096^2 matmul per
# GPU in fp32 and fp16 so the speed-up itself is on record, not just "it ran".
for i in range(n):
    dev = torch.device(f"cuda:{i}")
    a = torch.randn(4096, 4096, device=dev)
    b = torch.randn(4096, 4096, device=dev)
    for dtype in (torch.float32, torch.float16):
        a_, b_ = a.to(dtype), b.to(dtype)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(10):
            c = a_ @ b_
        torch.cuda.synchronize(dev)
        dt = (time.perf_counter() - t0) / 10
        finite = bool(torch.isfinite(c).all())
        print(f"gpu{i} {torch.cuda.get_device_name(i)} {str(dtype):14s} {dt*1e3:7.2f} ms/matmul finite={finite}")
        ok = ok and finite

# The actual mixed-precision path: autocast + GradScaler through a backward pass.
model = torch.nn.Linear(1024, 1024).cuda()
scaler = torch.amp.GradScaler("cuda")
x = torch.randn(256, 1024, device="cuda")
with torch.autocast("cuda", dtype=torch.float16):
    loss = model(x).float().pow(2).mean()
scaler.scale(loss).backward()
grad_finite = bool(torch.isfinite(model.weight.grad).all())
print("autocast_fp16_backward finite_grad", grad_finite, "loss", float(loss.detach()))
ok = ok and grad_finite

print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
