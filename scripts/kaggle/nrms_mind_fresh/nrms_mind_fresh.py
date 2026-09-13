"""P3.2 · NRMS + freshness term on MIND (SPEC.md §15): the P3.1 baseline kernel with the variant
plugged in — same recipe, seeds, epochs and data as ledger a2-nrms-mind v5; the only change is
`score = user·news + g(freshness)`. Writes `nrms_fresh` (ablation row 2) and `nrms_fresh_masked`
(row 3: same weights, term switched off at inference).

Push:  ./scripts/kaggle/alt_account.sh push scripts/kaggle/nrms_mind_fresh --accelerator NvidiaTeslaT4   (C-027: alt account)
Logs:  .venv/bin/python scripts/kaggle/ledger.py a2-nrms-mind-fresh vN --account alt
Files: KAGGLE_CONFIG_DIR=~/.kaggle/alt .venv/bin/kaggle kernels output aayushpandey602/a2-nrms-mind-fresh -p data/scores/_kaggle/mind_fresh --page-size 200 --file-pattern 'out/scores/.*'

Modes:
  DEMO_CHECK=True   MINDdemo (from huggingface.co/datasets/Recommenders/MIND, ungated), 1 epoch.
                    Pushed twice: identical group_auc to every digit = determinism oracle.
  DEMO_CHECK=False  the quick-start recipe on MINDsmall: 5 epochs, bs 32, history 50, npratio 4,
                    GloVe-300d from MINDsmall_utils.zip, title 30, 20x20 heads, attn 200, Adam 1e-4,
                    dropout 0.2 (nrms.yaml), seed 42. Data from the private dataset
                    aayushpandey18602/mind-small-official (official zips, hashes in the repo).
Departures from examples/00_quick_start/nrms_MIND.ipynb, each printed:
  * op determinism on (tf.config.experimental.enable_op_determinism) before the v1 session, and
    Python's `random` seeded (the package leaves it unseeded; it draws the negatives);
  * runs under tf-keras (TF_USE_LEGACY_KERAS=1) on TF 2.20, the package vendored --no-deps;
  * after training, run_fast_eval's per-impression predictions are written as the scores file
    (impr_index is the 0-based behaviors.tsv row = imp_row, proven in the U4 smoke), and the
    metrics are recomputed with our src/eval/metrics and required to match cal_metric's rounded
    values.
Before training the kernel runs `pytest tests/test_nrms_fresh_model_mind.py` (additive identity,
per-candidate locality, iterator batch + mask) and stops if it fails.
Every stage prints "STAGE <name> ok <seconds>".
"""
import hashlib, glob, json, os, subprocess, sys, time
import datetime as dt
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
DEMO_CHECK = True
EPOCHS = 1 if DEMO_CHECK else 5
SEED, BS = 42, 32
REC_COMMIT = "0bb4b3690941ffb668118e31ccaf8a7d19f8212a"
REPO_URL, REPO_REF = "https://github.com/Anurag-1205/News_Recommendation_System", "a2-click-logs"
HF = "https://huggingface.co/datasets/Recommenders/MIND/resolve/main"
WORK = Path("/kaggle/working"); DATA = WORK / "mind"; OUT = WORK / "out"
# ---- overlay (unpushed commit) ------------------------------------------------------------------
OVERLAY_FROM = "99025733748d922e67b883e3f4b1fd414c6ea599"
OVERLAY = {
    "src/baselines/nrms_fresh_features.py": "IiIiRnJlc2huZXNzIGlucHV0cyBmb3IgdGhlIE5STVMgdmFyaWFudCAoU1BFQy5tZCDCpzE1LjIpLgoKVGhlIG51bWJlciB0aGUgbW9kZWwgc2VlcyBpcyB0aGUgcmVyYW5rZXIncyBvd24gYGZyZXNobmVzc19ob3Vyc2AgKMKnMTEuOCwgYGZyZXNobmVzc19iYXRjaGAsCnN0cmljdGx5IGJlZm9yZSB0LCBOYU4gd2hlbiB1bmtub3duKSwgdHJhbnNmb3JtZWQgdG8gdHdvIGlucHV0cyBwZXIgY2FuZGlkYXRlOgoKICAgIHggICAgICAgPSAobG9nMXAoZnJlc2huZXNzX2hvdXJzKSDiiJIgzrwpIC8gz4MgICAgICDOvCwgz4MgZml0dGVkIG9uIHRoZSBUUkFJTklORyBzcGxpdCdzIGtub3duIGNhbmRpZGF0ZXMKICAgIHVua25vd24gPSAxIGlmIGZyZXNobmVzc19ob3VycyBpcyBOYU4gZWxzZSAwIDsgIHggPSAwIHdoZW4gdW5rbm93bgoKYGZyZXNoX2Zyb21fZnJhbWVzYCBpcyB0aGUgcHVyZSBmdW5jdGlvbiAodGVzdGVkIGJ5IGhhbmQgb24gYSB0b3kgc3BsaXQpOyBgZnJlc2hfaW5wdXRzYCB3aXJlcwppdCB0byBhIHJlYWwgc3BsaXQgd2l0aCB0aGUgc2FtZSBmaXJzdC1rbm93biBzb3VyY2VzIHRoZSByZXJhbmtlciB1c2VzLCBzbyB0aGUgdHdvIHN5c3RlbXMKYWdyZWUgb24gd2hhdCAiZnJlc2giIG1lYW5zOyBgYXNfbGlzdHNgIGdpdmVzIHRoZSBsb2FkZXJzIGEgbGlzdCBjb2x1bW4gYWxpZ25lZCB3aXRoCmBhcnRpY2xlX2lkc19pbnZpZXdgLgoiIiIKZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKZnJvbSBkYXRhY2xhc3NlcyBpbXBvcnQgYXNkaWN0LCBkYXRhY2xhc3MKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCgppbXBvcnQgbnVtcHkgYXMgbnAKaW1wb3J0IHBvbGFycyBhcyBwbAoKZnJvbSBzcmMuZmVhdHVyZXMuYmVoYXZpb3VyYWwgaW1wb3J0IGZyZXNobmVzc19iYXRjaAoKQ09MVU1OUyA9IFsiaW1wX3JvdyIsICJjYW5kX3Bvc2l0aW9uIiwgImZyZXNobmVzc19ob3VycyIsICJ4IiwgInVua25vd24iXQoKCkBkYXRhY2xhc3MoZnJvemVuPVRydWUpCmNsYXNzIEZyZXNoU3RhdHM6CiAgICBtdTogZmxvYXQKICAgIHNpZ21hOiBmbG9hdAoKICAgIGRlZiB0b19kaWN0KHNlbGYpIC0+IGRpY3Q6CiAgICAgICAgcmV0dXJuIGFzZGljdChzZWxmKQoKCmRlZiBmcmVzaF9mcm9tX2ZyYW1lcyhiZWg6IHBsLkRhdGFGcmFtZSwgZmlyc3Rfa25vd246IHBsLkRhdGFGcmFtZSwgc3RhdHM6IEZyZXNoU3RhdHMsICosCiAgICAgICAgICAgICAgICAgICAgICB1bnRpbWVkX3RzPU5vbmUsIGtleTogc3RyID0gImltcF9yb3ciKSAtPiBwbC5EYXRhRnJhbWU6CiAgICAiIiJgYmVoYDogYGtleWAsIGB0YCwgYGFydGljbGVfaWRzX2ludmlld2AuIGBmaXJzdF9rbm93bmA6IGBhcnRpY2xlX2lkYCwgYHRzYCAowqcxMS44KS4KICAgIE9uZSByb3cgcGVyIGNhbmRpZGF0ZSwgc2xhdGUgb3JkZXIsIHdpdGggdGhlIHJhdyBob3VycyBhbmQgdGhlIHR3byBtb2RlbCBpbnB1dHMuCgogICAgYGtleWAgbXVzdCBpZGVudGlmeSBhICpyb3cqIG9mIGBiZWhgLiBGb3IgYSBzcGxpdCBpdCBpcyBgaW1wX3Jvd2A7IGZvciBhIHd1MjAxOS1zYW1wbGVkCiAgICB0cmFpbmluZyBmcmFtZSBpdCBtdXN0IGJlIGEgZnJlc2ggcm93IGluZGV4IOKAlCBzYW1wbGluZyB0dXJucyBhbiBpbXByZXNzaW9uIHdpdGggayBjbGlja3MgaW50bwogICAgayByb3dzIHRoYXQgc2hhcmUgYGltcF9yb3dgIChFQi1OZVJEIGRlbW8gcnVuIHYyIGRpZWQgb24gZXhhY3RseSB0aGF0KS4iIiIKICAgIHJlcSA9IChiZWguc2VsZWN0KHBsLmNvbChrZXkpLmFsaWFzKCJpbXBfcm93IiksICJ0IiwgYXJ0aWNsZV9pZD1wbC5jb2woImFydGljbGVfaWRzX2ludmlldyIpKQogICAgICAgICAgIC53aXRoX2NvbHVtbnMoY2FuZF9wb3NpdGlvbj1wbC5pbnRfcmFuZ2VzKDEsIHBsLmNvbCgiYXJ0aWNsZV9pZCIpLmxpc3QubGVuKCkgKyAxLCBkdHlwZT1wbC5JbnQ2NCkpCiAgICAgICAgICAgLmV4cGxvZGUoImFydGljbGVfaWQiLCAiY2FuZF9wb3NpdGlvbiIsIGVtcHR5X2FzX251bGw9RmFsc2UpKQogICAgb3V0ID0gZnJlc2huZXNzX2JhdGNoKGZpcnN0X2tub3duLCByZXEsIHVudGltZWRfdHM9dW50aW1lZF90cykKICAgIHggPSAoKHBsLmNvbCgiZnJlc2huZXNzX2hvdXJzIikubG9nMXAoKSAtIHN0YXRzLm11KSAvIHN0YXRzLnNpZ21hKQogICAgcmV0dXJuIChvdXQud2l0aF9jb2x1bW5zKHVua25vd249cGwuY29sKCJmcmVzaG5lc3NfaG91cnMiKS5pc19uYW4oKS5jYXN0KHBsLkludDgpKQogICAgICAgICAgICAud2l0aF9jb2x1bW5zKHg9cGwud2hlbihwbC5jb2woInVua25vd24iKSA9PSAxKS50aGVuKDAuMCkub3RoZXJ3aXNlKHgpLmNhc3QocGwuRmxvYXQzMikpCiAgICAgICAgICAgIC5zZWxlY3QoQ09MVU1OUykuc29ydCgiaW1wX3JvdyIsICJjYW5kX3Bvc2l0aW9uIikpCgoKZGVmIGZpdF9zdGF0cyhmZWF0czogcGwuRGF0YUZyYW1lKSAtPiBGcmVzaFN0YXRzOgogICAgIiIizrwsIM+DIG9mIGxvZzFwKGhvdXJzKSBvdmVyIHRoZSBrbm93biBjYW5kaWRhdGVzIG9mIGEgZnJhbWUgKHVzZSB0aGUgVFJBSU5JTkcgc3BsaXQpLiIiIgogICAgdiA9IG5wLmxvZzFwKGZlYXRzLmZpbHRlcihwbC5jb2woInVua25vd24iKSA9PSAwKVsiZnJlc2huZXNzX2hvdXJzIl0udG9fbnVtcHkoKSkKICAgIHJldHVybiBGcmVzaFN0YXRzKG11PWZsb2F0KHYubWVhbigpKSwgc2lnbWE9ZmxvYXQodi5zdGQoKSkpCgoKZGVmIGFzX2xpc3RzKGJlaDogcGwuRGF0YUZyYW1lLCBmZWF0czogcGwuRGF0YUZyYW1lLCAqLCBrZXk6IHN0ciA9ICJpbXBfcm93IikgLT4gcGwuRGF0YUZyYW1lOgogICAgIiIiYChrZXksIGZyZXNoX2ludmlldylgOiBwZXIgcm93IG9mIGBiZWhgLCBhIGxpc3Qgb2YgW3gsIHVua25vd25dIHBhaXJzIGluIHNsYXRlIG9yZGVyLgogICAgYGZlYXRzYCBpcyBgZnJlc2hfZnJvbV9mcmFtZXMoYmVoLCDigKYsIGtleT1rZXkpYCwgd2hvc2UgYGltcF9yb3dgIGNvbHVtbiBob2xkcyB0aGF0IGtleTsKICAgIGBiZWhba2V5XWAgbXVzdCBiZSB1bmlxdWUgKGEgZnJlc2ggcm93IGluZGV4IG9uIGEgc2FtcGxlZCBmcmFtZSkuIiIiCiAgICBpZiBiZWhba2V5XS5uX3VuaXF1ZSgpICE9IGJlaC5oZWlnaHQ6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcihmIntrZXl9IGlzIG5vdCB1bmlxdWUgcGVyIHJvdzsgdXNlIGEgcm93IGluZGV4IGFzIHRoZSBrZXkgb24gc2FtcGxlZCBmcmFtZXMiKQogICAgcGFpcnMgPSAoZmVhdHMuc29ydCgiaW1wX3JvdyIsICJjYW5kX3Bvc2l0aW9uIikKICAgICAgICAgICAgIC53aXRoX2NvbHVtbnMocGFpcj1wbC5jb25jYXRfbGlzdChwbC5jb2woIngiKS5jYXN0KHBsLkZsb2F0MzIpLCBwbC5jb2woInVua25vd24iKS5jYXN0KHBsLkZsb2F0MzIpKSkKICAgICAgICAgICAgIC5ncm91cF9ieSgiaW1wX3JvdyIsIG1haW50YWluX29yZGVyPVRydWUpLmFnZyhmcmVzaF9pbnZpZXc9cGwuY29sKCJwYWlyIikpCiAgICAgICAgICAgICAucmVuYW1lKHsiaW1wX3JvdyI6IGtleX0pKQogICAgb3V0ID0gYmVoLnNlbGVjdChrZXkpLmpvaW4ocGFpcnMsIG9uPWtleSwgaG93PSJsZWZ0IiwgbWFpbnRhaW5fb3JkZXI9ImxlZnQiKQogICAgbGVucyA9IG91dFsiZnJlc2hfaW52aWV3Il0ubGlzdC5sZW4oKS5maWxsX251bGwoMCkKICAgIGlmIG5vdCAobGVucyA9PSBiZWhbImFydGljbGVfaWRzX2ludmlldyJdLmxpc3QubGVuKCkpLmFsbCgpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoImZyZXNoX2ludmlldyBsZW5ndGhzIGRvIG5vdCBtYXRjaCBhcnRpY2xlX2lkc19pbnZpZXciKQogICAgcmV0dXJuIG91dAoKCmRlZiBfZWJuZXJkX2ZyYW1lcyhzcGxpdDogc3RyLCBsaW1pdDogaW50IHwgTm9uZSk6CiAgICBmcm9tIHNyYy5iYXNlbGluZXMubnJtc19kYXRhIGltcG9ydCBlYm5lcmRfYmVoYXZpb3JzCiAgICBmcm9tIHNyYy5yZXJhbmsuZWJuZXJkIGltcG9ydCBsb2FkX2FydGljbGVzCiAgICBiZWggPSBlYm5lcmRfYmVoYXZpb3JzKFBhdGgoImRhdGEvaW50ZXJpbS9lYm5lcmQiKSAvIHNwbGl0LCBoaXN0b3J5X3NpemU9MSwgbGltaXQ9bGltaXQpCiAgICBiZWggPSBiZWguc2VsZWN0KCJpbXBfcm93IiwgdD1wbC5jb2woImltcHJlc3Npb25fdGltZSIpLCBhcnRpY2xlX2lkc19pbnZpZXc9cGwuY29sKCJhcnRpY2xlX2lkc19pbnZpZXciKSkKICAgIGZpcnN0X2tub3duID0gbG9hZF9hcnRpY2xlcygpLnNlbGVjdCgiYXJ0aWNsZV9pZCIsIHRzPXBsLmNvbCgicHVibGlzaGVkX3RpbWUiKSkKICAgIHJldHVybiBiZWgsIGZpcnN0X2tub3duLCBOb25lCgoKZGVmIF9taW5kX2ZyYW1lcyhzcGxpdDogc3RyLCBsaW1pdDogaW50IHwgTm9uZSk6CiAgICBmcm9tIHNyYy5yZXJhbmsubWluZCBpbXBvcnQgREFUQVNFVF9TVEFSVCwgZmlyc3Rfc2lnaHRpbmdzLCBsb2FkX2JlaGF2aW9ycwogICAgdHJhaW4sIGRldiA9IGxvYWRfYmVoYXZpb3JzKCJNSU5Ec21hbGxfdHJhaW4iKSwgbG9hZF9iZWhhdmlvcnMoIk1JTkRzbWFsbF9kZXYiKQogICAgc2lnaHRpbmdzID0gZmlyc3Rfc2lnaHRpbmdzKFt0cmFpbiwgZGV2XSkgICAgICAgICAgIyBzdHJpY3QgPCB0IGtlZXBzIGRldiBzaWdodGluZ3Mgb3V0IG9mIHRyYWluIHJvd3MKICAgIGJlaCA9IHsiTUlORHNtYWxsX3RyYWluIjogdHJhaW4sICJNSU5Ec21hbGxfZGV2IjogZGV2fVtzcGxpdF0KICAgIGlmIGxpbWl0OgogICAgICAgIGJlaCA9IGJlaC5oZWFkKGxpbWl0KQogICAgYmVoID0gYmVoLnNlbGVjdCgiaW1wX3JvdyIsICJ0IiwgYXJ0aWNsZV9pZHNfaW52aWV3PXBsLmNvbCgiY2FuZGlkYXRlcyIpKQogICAgcmV0dXJuIGJlaCwgc2lnaHRpbmdzLCBEQVRBU0VUX1NUQVJUCgoKZGVmIGZyZXNoX2lucHV0cyhkYXRhc2V0OiBzdHIsIHNwbGl0OiBzdHIsIHN0YXRzOiBGcmVzaFN0YXRzIHwgTm9uZSwgKiwgbGltaXQ6IGludCB8IE5vbmUgPSBOb25lKSAtPiBwbC5EYXRhRnJhbWU6CiAgICAiIiJGcmVzaG5lc3MgaW5wdXRzIGZvciBldmVyeSBjYW5kaWRhdGUgb2YgYSByZWFsIHNwbGl0LiBgc3RhdHM9Tm9uZWAgZml0cyDOvCwgz4Mgb24gdGhpcyBzcGxpdAogICAgKGRvIHRoYXQgb24gdGhlIHRyYWluaW5nIHNwbGl0IG9ubHksIHRoZW4gcGFzcyB0aGUgcmVzdWx0IGZvciB0aGUgZXZhbHVhdGlvbiBzcGxpdCkuIiIiCiAgICBiZWgsIGZpcnN0X2tub3duLCB1bnRpbWVkID0geyJlYm5lcmQiOiBfZWJuZXJkX2ZyYW1lcywgIm1pbmQiOiBfbWluZF9mcmFtZXN9W2RhdGFzZXRdKHNwbGl0LCBsaW1pdCkKICAgIHJhdyA9IGZyZXNoX2Zyb21fZnJhbWVzKGJlaCwgZmlyc3Rfa25vd24sIEZyZXNoU3RhdHMoMC4wLCAxLjApLCB1bnRpbWVkX3RzPXVudGltZWQpCiAgICBpZiBzdGF0cyBpcyBOb25lOgogICAgICAgIHN0YXRzID0gZml0X3N0YXRzKHJhdykKICAgIHJldHVybiBmcmVzaF9mcm9tX2ZyYW1lcyhiZWgsIGZpcnN0X2tub3duLCBzdGF0cywgdW50aW1lZF90cz11bnRpbWVkKQo=",
    "tests/test_nrms_fresh_model_mind.py": "IiIiTW9kZWwtbGV2ZWwgb3JhY2xlcyBmb3IgdGhlIE1JTkQgKHJlY29tbWVuZGVycykgZnJlc2huZXNzIHZhcmlhbnQgKFNQRUMubWQgwqcxNS40KS4gTmVlZAp0Zi1rZXJhcyAoVEZfVVNFX0xFR0FDWV9LRVJBUz0xKSwgdGhlIHJlY29tbWVuZGVycyBwYWNrYWdlIG9uIHN5cy5wYXRoIGFuZCB0aGUgTUlORCB1dGlscwooYE5STVNfWUFNTGAsIGBOUk1TX0VNQmAsIGBOUk1TX1dESUNUYCwgYE5STVNfVURJQ1RgIGVudiB2YXJzKSDigJQgc28gdGhleSBza2lwIG9uIHRoZSBsYXB0b3AgYW5kCnJ1biBpbnNpZGUgdGhlIEthZ2dsZSBrZXJuZWwgYmVmb3JlIHRyYWluaW5nLiIiIgppbXBvcnQgb3MKaW1wb3J0IHN5cwpmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKCmltcG9ydCBudW1weSBhcyBucAppbXBvcnQgcHl0ZXN0Cgp0ZiA9IHB5dGVzdC5pbXBvcnRvcnNraXAoInRlbnNvcmZsb3ciKQpmb3IgcCBpbiAoImV4dGVybmFsL3JlY29tbWVuZGVycyIsICIva2FnZ2xlL3dvcmtpbmcvcmVjb21tZW5kZXJzIik6CiAgICBpZiBQYXRoKHApLmV4aXN0cygpIGFuZCBwIG5vdCBpbiBzeXMucGF0aDoKICAgICAgICBzeXMucGF0aC5pbnNlcnQoMCwgcCkKcmVjID0gcHl0ZXN0LmltcG9ydG9yc2tpcCgicmVjb21tZW5kZXJzIikKaWYgbm90IGFsbChvcy5lbnZpcm9uLmdldChrKSBmb3IgayBpbiAoIk5STVNfWUFNTCIsICJOUk1TX0VNQiIsICJOUk1TX1dESUNUIiwgIk5STVNfVURJQ1QiKSk6CiAgICBweXRlc3Quc2tpcCgiTUlORCB1dGlscyBlbnYgdmFycyBub3Qgc2V0IiwgYWxsb3dfbW9kdWxlX2xldmVsPVRydWUpCgpmcm9tIHJlY29tbWVuZGVycy5tb2RlbHMubmV3c3JlYy5uZXdzcmVjX3V0aWxzIGltcG9ydCBwcmVwYXJlX2hwYXJhbXMgICAjIG5vcWE6IEU0MDIKZnJvbSByZWNvbW1lbmRlcnMubW9kZWxzLm5ld3NyZWMubW9kZWxzLm5ybXMgaW1wb3J0IE5STVNNb2RlbCAgICAgICAgICAgICMgbm9xYTogRTQwMgpmcm9tIHJlY29tbWVuZGVycy5tb2RlbHMubmV3c3JlYy5pby5taW5kX2l0ZXJhdG9yIGltcG9ydCBNSU5ESXRlcmF0b3IgICAgIyBub3FhOiBFNDAyCmZyb20gc3JjLmJhc2VsaW5lcy5ucm1zX2ZyZXNoX3JlYyBpbXBvcnQgRlJFU0hfRElNLCBNSU5ERnJlc2hJdGVyYXRvciwgTlJNU0ZyZXNoTW9kZWwgICMgbm9xYTogRTQwMgoKCmRlZiBfaHBhcmFtcygpOgogICAgcmV0dXJuIHByZXBhcmVfaHBhcmFtcyhvcy5lbnZpcm9uWyJOUk1TX1lBTUwiXSwgd29yZEVtYl9maWxlPW9zLmVudmlyb25bIk5STVNfRU1CIl0sIHdvcmREaWN0X2ZpbGU9b3MuZW52aXJvblsiTlJNU19XRElDVCJdLAogICAgICAgICAgICAgICAgICAgICAgICAgICB1c2VyRGljdF9maWxlPW9zLmVudmlyb25bIk5STVNfVURJQ1QiXSwgYmF0Y2hfc2l6ZT00LCBlcG9jaHM9MSwgc2hvd19zdGVwPTEwKio5KQoKCmRlZiBfaW5wdXRzKHJuZywgaHAsIGJhdGNoPTMpOgogICAgdm9jYWIgPSAxMDAwCiAgICBoaXMgPSBybmcuaW50ZWdlcnMoMSwgdm9jYWIsIHNpemU9KGJhdGNoLCBocC5oaXNfc2l6ZSwgaHAudGl0bGVfc2l6ZSkpLmFzdHlwZSgiaW50MzIiKQogICAgcHJlZCA9IHJuZy5pbnRlZ2VycygxLCB2b2NhYiwgc2l6ZT0oYmF0Y2gsIGhwLm5wcmF0aW8gKyAxLCBocC50aXRsZV9zaXplKSkuYXN0eXBlKCJpbnQzMiIpCiAgICBmcmVzaCA9IG5wLnN0YWNrKFtybmcuc3RhbmRhcmRfbm9ybWFsKChiYXRjaCwgaHAubnByYXRpbyArIDEpKSwgcm5nLmludGVnZXJzKDAsIDIsIChiYXRjaCwgaHAubnByYXRpbyArIDEpKV0sIC0xKS5hc3R5cGUoImZsb2F0MzIiKQogICAgcmV0dXJuIGhpcywgcHJlZCwgZnJlc2gKCgpkZWYgdGVzdF9hZGRpdGl2ZV9pZGVudGl0eV9nX3plcm9fcmVwcm9kdWNlc19iYXNlbGluZSgpOgogICAgaHAgPSBfaHBhcmFtcygpCiAgICBiYXNlID0gTlJNU01vZGVsKGhwLCBNSU5ESXRlcmF0b3IsIHNlZWQ9NDIpCiAgICB2YXIgPSBOUk1TRnJlc2hNb2RlbChocCwgTUlOREZyZXNoSXRlcmF0b3IsIHNlZWQ9NDIpCiAgICB2YXIuY29weV9lbmNvZGVyc19mcm9tKGJhc2UpOyB2YXIuemVyb19nKCkKICAgIGhpcywgcHJlZCwgZnJlc2ggPSBfaW5wdXRzKG5wLnJhbmRvbS5kZWZhdWx0X3JuZygwKSwgaHApCiAgICBucC50ZXN0aW5nLmFzc2VydF9hbGxjbG9zZSh2YXIubW9kZWwucHJlZGljdChbaGlzLCBwcmVkLCBmcmVzaF0pLCBiYXNlLm1vZGVsLnByZWRpY3QoW2hpcywgcHJlZF0pLCBhdG9sPTFlLTYpCiAgICBucC50ZXN0aW5nLmFzc2VydF9hbGxjbG9zZSh2YXIuc2NvcmVyLnByZWRpY3QoW2hpcywgcHJlZFs6LCA6MV0sIGZyZXNoWzosIDoxXV0pLCBiYXNlLnNjb3Jlci5wcmVkaWN0KFtoaXMsIHByZWRbOiwgOjFdXSksIGF0b2w9MWUtNikKCgpkZWYgdGVzdF9mcmVzaG5lc3NfdGVybV9pc19wZXJfY2FuZGlkYXRlKCk6CiAgICBocCA9IF9ocGFyYW1zKCkKICAgIHZhciA9IE5STVNGcmVzaE1vZGVsKGhwLCBNSU5ERnJlc2hJdGVyYXRvciwgc2VlZD00MikKICAgIGhpcywgcHJlZCwgZnJlc2ggPSBfaW5wdXRzKG5wLnJhbmRvbS5kZWZhdWx0X3JuZygxKSwgaHAsIGJhdGNoPTEpCiAgICBwMCA9IHZhci5tb2RlbC5wcmVkaWN0KFtoaXMsIHByZWQsIGZyZXNoXSkKICAgIGZyZXNoMiA9IGZyZXNoLmNvcHkoKTsgZnJlc2gyWzAsIDIsIDBdICs9IDMuMAogICAgcDEgPSB2YXIubW9kZWwucHJlZGljdChbaGlzLCBwcmVkLCBmcmVzaDJdKQogICAgYXNzZXJ0IG5vdCBucC5hbGxjbG9zZShwMCwgcDEpCiAgICBnID0gdmFyLmdfdmFsdWVzKG5wLmFycmF5KFtbMC4wLCAxLjBdLCBbMC4wLCAxLjBdXSwgZHR5cGU9ImZsb2F0MzIiKSkKICAgIGFzc2VydCBnLnNoYXBlID09ICgyLCkgYW5kIGdbMF0gPT0gZ1sxXQoKCmRlZiB0ZXN0X2l0ZXJhdG9yX3lpZWxkc19mcmVzaF9iYXRjaF9hbmRfbWFzayh0bXBfcGF0aCk6CiAgICAiIiJBIDItbGluZSBiZWhhdmlvcnMgZmlsZTogdGhlIGZyZXNoIGJhdGNoIGxpbmVzIHVwIHdpdGggdGhlIHNhbXBsZWQgY2FuZGlkYXRlcy4iIiIKICAgIGhwID0gX2hwYXJhbXMoKQogICAgbmV3cyA9IHRtcF9wYXRoIC8gIm5ld3MudHN2IjsgYmVoID0gdG1wX3BhdGggLyAiYmVoYXZpb3JzLnRzdiIKICAgIG5ld3Mud3JpdGVfdGV4dCgiXG4iLmpvaW4oZiJOe2l9XHRzcG9ydHNcdGZvb3RiYWxsXHR0aXRsZSB7aX0gd29yZHMgaGVyZVx0YWJzXHR1cmxcdFtdXHRbXSIgZm9yIGkgaW4gcmFuZ2UoMSwgOCkpICsgIlxuIikKICAgIGJlaC53cml0ZV90ZXh0KCIxXHRVMVx0MTEvMTUvMjAxOSAxMjowMDowMCBQTVx0TjEgTjJcdE4zLTEgTjQtMCBONS0wIE42LTAgTjctMFxuIgogICAgICAgICAgICAgICAgICAgIjJcdFUyXHQxMS8xNS8yMDE5IDEyOjA1OjAwIFBNXHROMlx0TjQtMCBONS0xIE42LTBcbiIpCiAgICBpdCA9IE1JTkRGcmVzaEl0ZXJhdG9yKGhwLCBucHJhdGlvPWhwLm5wcmF0aW8sIGNvbF9zcGxpdGVyPSJcdCIpCiAgICBpdC5mcmVzaF9ieV9yb3cgPSBbe2YiTntpfSI6ICgwLjEgKiBpLCAwLjApIGZvciBpIGluIHJhbmdlKDMsIDgpfSwgeyJONCI6ICgwLjQsIDAuMCksICJONSI6ICgwLjUsIDAuMCl9XSAgICMgTjYgdW5rbm93biBpbiByb3cgMgogICAgYmF0Y2hlcyA9IGxpc3QoaXQubG9hZF9kYXRhX2Zyb21fZmlsZShzdHIobmV3cyksIHN0cihiZWgpKSkKICAgIGIgPSBiYXRjaGVzWzBdCiAgICBhc3NlcnQgYlsiY2FuZGlkYXRlX2ZyZXNoX2JhdGNoIl0uc2hhcGUgPT0gKDIsIGhwLm5wcmF0aW8gKyAxLCBGUkVTSF9ESU0pCiAgICAjIGZpcnN0IHNhbXBsZSBvZiBlYWNoIGxpbmUgaXMgaXRzIHBvc2l0aXZlOiByb3cgMCDihpIgTjMgKDAuMyksIHJvdyAxIOKGkiBONSAoMC41KTsgb3JkZXIgbWF5IGJlIHNodWZmbGVkCiAgICBmaXJzdHMgPSBzb3J0ZWQoYlsiY2FuZGlkYXRlX2ZyZXNoX2JhdGNoIl1bOiwgMCwgMF0ucm91bmQoMykudG9saXN0KCkpCiAgICBhc3NlcnQgZmlyc3RzID09IFswLjMsIDAuNV0KICAgIGl0Lm1hc2sgPSBUcnVlCiAgICBibSA9IG5leHQoaXQubG9hZF9kYXRhX2Zyb21fZmlsZShzdHIobmV3cyksIHN0cihiZWgpKSkKICAgIGFzc2VydCAoYm1bImNhbmRpZGF0ZV9mcmVzaF9iYXRjaCJdLnJlc2hhcGUoLTEsIEZSRVNIX0RJTSkgPT0gbnAuYXJyYXkoWzAuMCwgMS4wXSwgZHR5cGU9ImZsb2F0MzIiKSkuYWxsKCkK"
}
# ---- end overlay ----

T0 = time.time()


def stage(name): print(f"STAGE {name} ok {time.time() - T0:.0f}s", flush=True)
def sh(cmd): print("+", cmd, flush=True); subprocess.run(cmd, shell=True, check=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


print("KNOBS", json.dumps({"DEMO_CHECK": DEMO_CHECK, "EPOCHS": EPOCHS, "SEED": SEED, "BS": BS}))

# ---- 1. environment + code -------------------------------------------------------------------------
sh("pip install -q 'tf-keras~=2.20.0' 'retrying>=1.3.4' polars==1.43.2")
sh(f"git clone -q https://github.com/recommenders-team/recommenders {WORK}/recommenders && cd {WORK}/recommenders && git checkout -q {REC_COMMIT} && git rev-parse HEAD")
sh(f"git clone -q --branch {REPO_REF} {REPO_URL} {WORK}/repo && cd {WORK}/repo && git rev-parse HEAD")
REPO_COMMIT = subprocess.check_output(["git", "-C", str(WORK / "repo"), "rev-parse", "HEAD"], text=True).strip()
OVERLAY_NOTE = None
if "OVERLAY" in globals() and OVERLAY:      # files from a commit the laptop could not push yet (scripts/kaggle/overlay.py)
    import base64, hashlib
    for rel, b64 in OVERLAY.items():
        data = base64.b64decode(b64); (WORK / "repo" / rel).write_bytes(data)
        print("OVERLAY", rel, "sha256", hashlib.sha256(data).hexdigest()[:12])
    print("OVERLAY_FROM", OVERLAY_FROM)
    OVERLAY_NOTE = {"from_commit": OVERLAY_FROM, "files": sorted(OVERLAY)}
sys.path[:0] = [str(WORK / "recommenders"), str(WORK / "repo")]
import numpy as np, polars as pl, tensorflow as tf
tf.config.experimental.enable_op_determinism()
print("tf", tf.__version__, "tf.keras", tf.keras.__name__, "polars", pl.__version__, "GPUs", [g.name for g in tf.config.list_physical_devices("GPU")])
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.deeprec.deeprec_utils import cal_metric
from src.baselines.nrms_data import scores_frame
from src.baselines.nrms_fresh_features import FreshStats, fit_stats, fresh_inputs
from src.baselines.nrms_fresh_rec import MINDFreshIterator, NRMSFreshModel
from src.eval.metrics import per_impression_metrics
from src.rerank.common import evaluate as bootstrap_evaluate, impression_rows
stage("imports")

# ---- 2. data ----------------------------------------------------------------------------------------
if DEMO_CHECK:
    for name, sub in [("MINDdemo_train", "train"), ("MINDdemo_dev", "valid"), ("MINDdemo_utils", "utils")]:
        sh(f"mkdir -p {DATA}/{sub} && wget -q {HF}/{name}.zip -O {DATA}/{name}.zip && unzip -q -o {DATA}/{name}.zip -d {DATA}/{sub}")
        print(f"{name}.zip sha256", sha(DATA / f"{name}.zip"))
else:
    for name, sub in [("MINDsmall_train", "train"), ("MINDsmall_dev", "valid")]:
        hits = sorted(glob.glob(f"/kaggle/input/**/{name}/**/behaviors.tsv", recursive=True))
        assert hits, f"no behaviors.tsv for {name} under /kaggle/input"
        src = Path(hits[0]).parent
        for tsv in ("behaviors.tsv", "news.tsv"):
            print(f"{name}/{tsv} sha256", sha(src / tsv))
        sh(f"mkdir -p {DATA}/{sub} && cp {src}/behaviors.tsv {src}/news.tsv {DATA}/{sub}/")
    sh(f"mkdir -p {DATA}/utils && wget -q {HF}/MINDsmall_utils.zip -O {DATA}/MINDsmall_utils.zip && unzip -q -o {DATA}/MINDsmall_utils.zip -d {DATA}/utils")
    print("MINDsmall_utils.zip sha256", sha(DATA / "MINDsmall_utils.zip"))
f = {k: str(DATA / p) for k, p in {"train_news": "train/news.tsv", "train_beh": "train/behaviors.tsv",
     "valid_news": "valid/news.tsv", "valid_beh": "valid/behaviors.tsv", "emb": "utils/embedding.npy",
     "udict": "utils/uid2index.pkl", "wdict": "utils/word_dict.pkl", "yaml": "utils/nrms.yaml"}.items()}
# the repo's loaders read data/interim/mind/<split>/<split>/*.tsv relative to the repo root
if not DEMO_CHECK:
    for name, sub in [("MINDsmall_train", "train"), ("MINDsmall_dev", "valid")]:
        sh(f"mkdir -p {WORK}/repo/data/interim/mind/{name}/{name} && cp {DATA}/{sub}/behaviors.tsv {DATA}/{sub}/news.tsv {WORK}/repo/data/interim/mind/{name}/{name}/")
os.environ.update(NRMS_YAML=f["yaml"], NRMS_EMB=f["emb"], NRMS_WDICT=f["wdict"], NRMS_UDICT=f["udict"])
sh(f"cd {WORK}/repo && PYTHONPATH=.:{WORK}/recommenders TF_USE_LEGACY_KERAS=1 python -m pytest tests/test_nrms_fresh_model_mind.py -q -p no:cacheprovider")
stage("data+oracle")

# ---- 2b. freshness table: the reranker's first-seen feature, standardised on the train split ----------
def fresh_by_row(feats, beh_frame):
    """list indexed by behaviors row → {news_id: (x, unknown)}, from an (imp_row, cand_position, x, unknown) frame."""
    ids = beh_frame.select("imp_row", "candidates").explode("candidates").with_columns(
        cand_position=pl.int_range(1, pl.len() + 1).over("imp_row").cast(pl.Int64))
    j = feats.join(ids, on=["imp_row", "cand_position"], how="left")
    out = [dict() for _ in range(beh_frame.height)]
    for r, nid, x, u in zip(j["imp_row"].to_list(), j["candidates"].to_list(), j["x"].to_list(), j["unknown"].to_list()):
        out[r][nid] = (float(x), float(u))
    return out

if DEMO_CHECK:
    # MINDdemo is not in the repo's layout; use per-split first-seen from the demo files themselves
    import os as _os; _os.chdir(WORK / "repo")
    from src.rerank.mind import first_sightings, load_behaviors, DATASET_START
    from src.baselines.nrms_fresh_features import fresh_from_frames
    for name, sub in [("MINDsmall_train", "train"), ("MINDsmall_dev", "valid")]:      # demo files under the small names
        sh(f"mkdir -p {WORK}/repo/data/interim/mind/{name}/{name} && cp {DATA}/{sub}/behaviors.tsv {DATA}/{sub}/news.tsv {WORK}/repo/data/interim/mind/{name}/{name}/")
else:
    import os as _os; _os.chdir(WORK / "repo")
    from src.rerank.mind import load_behaviors
f_tr = fresh_inputs("mind", "MINDsmall_train", None); STATS = fit_stats(f_tr)
f_tr = fresh_inputs("mind", "MINDsmall_train", STATS); f_ev = fresh_inputs("mind", "MINDsmall_dev", STATS)
print("FRESH_STATS", json.dumps(STATS.to_dict()), "| train unknown %.3f%% | dev unknown %.3f%%" % (100 * f_tr["unknown"].mean(), 100 * f_ev["unknown"].mean()))
FRESH_TRAIN = fresh_by_row(f_tr, load_behaviors("MINDsmall_train").select("imp_row", "candidates"))
FRESH_DEV = fresh_by_row(f_ev, load_behaviors("MINDsmall_dev").select("imp_row", "candidates"))
_os.chdir(WORK)
stage("fresh")

# ---- 3. train ---------------------------------------------------------------------------------------
hparams = prepare_hparams(f["yaml"], wordEmb_file=f["emb"], wordDict_file=f["wdict"], userDict_file=f["udict"],
                          batch_size=BS, epochs=EPOCHS, show_step=100000)
print("HPARAMS", str(hparams)[:600])
model = NRMSFreshModel(hparams, MINDFreshIterator, seed=SEED)
model.train_iterator.fresh_by_row = FRESH_TRAIN
model.test_iterator.fresh_by_row = FRESH_DEV
print("params", model.model.count_params())
# The package seeds TF and numpy (base_model.__init__) but not Python's `random`, which
# `newsrec_utils.newsample` uses to draw the npratio negatives: demo twin runs v1/v2 differed
# (AUC 0.5805 vs 0.5784). Seeding it here makes the sampling sequence reproducible.
import random; random.seed(SEED); np.random.seed(SEED)
t = time.time()
model.fit(f["train_news"], f["train_beh"], f["valid_news"], f["valid_beh"])
train_seconds = time.time() - t
print(f"TRAIN_SECONDS {train_seconds:.0f} (epochs={EPOCHS}, incl. the per-epoch eval)")
WEIGHTS = OUT / "weights"; WEIGHTS.mkdir(parents=True, exist_ok=True)
model.model.save_weights(str(WEIGHTS / "nrms_fresh_ckpt"))
stage("train")

# ---- 4+5. score every dev impression, full slate, and evaluate: twice (row 2, row 3 masked) -------------
rows = [line.rstrip("\n").split("\t") for line in open(f["valid_beh"])]
beh = pl.DataFrame({
    "imp_row": pl.Series(range(len(rows)), dtype=pl.UInt32),
    "impression_id": pl.Series([int(r[0]) for r in rows], dtype=pl.Int64),
    "article_ids_inview": pl.Series([[x.split("-")[0] for x in r[4].split()] for r in rows], dtype=pl.List(pl.Utf8)),
    "labels": pl.Series([[int(x.split("-")[1]) for x in r[4].split()] for r in rows], dtype=pl.List(pl.Int8)),
})
lab = (beh.select("imp_row", "labels").explode("labels")
       .with_columns(cand_position=pl.int_range(1, pl.len() + 1).over("imp_row").cast(pl.Int64)))
results = {}
for system, mask in [("nrms_fresh", False), ("nrms_fresh_masked", True)]:
    model.test_iterator.mask = mask
    t = time.time()
    impr_idx, labels, preds = model.run_fast_eval(f["valid_news"], f["valid_beh"])
    score_seconds = time.time() - t
    assert list(impr_idx) == list(range(beh.height)), "impr_index is not the file row"
    assert [list(map(int, l)) for l in labels] == beh["labels"].to_list()
    sf = scores_frame(beh.select("imp_row", "impression_id", "article_ids_inview"), [list(map(float, p)) for p in preds])
    assert sf.height == int(beh["article_ids_inview"].list.len().sum()) and sf["imp_row"].n_unique() == beh.height
    SC = OUT / "scores" / "mind" / ("MINDdemo_dev" if DEMO_CHECK else "MINDsmall_dev"); SC.mkdir(parents=True, exist_ok=True)
    sf.write_parquet(SC / f"{system}.parquet")
    manifest = {"dataset": "mind", "split": "MINDdemo_dev" if DEMO_CHECK else "MINDsmall_dev", "system": system, "framing": "in-impression",
                "variant": "nrms + freshness term (SPEC §15)", "fresh_masked_at_inference": mask, "fresh_stats": STATS.to_dict(),
                "n_rows": sf.height, "n_impressions": beh.height, "repo_commit": REPO_COMMIT, "overlay": OVERLAY_NOTE, "recommenders_commit": REC_COMMIT,
                "seeds": {"model": SEED}, "epochs": EPOCHS, "history_size": hparams.his_size, "npratio": hparams.npratio,
                "word_embeddings": "GloVe-300d (MIND utils embedding.npy)", "precision": "float32", "determinism": True,
                "train_seconds": round(train_seconds), "score_seconds": round(score_seconds),
                "command": "./scripts/kaggle/alt_account.sh push scripts/kaggle/nrms_mind_fresh --accelerator NvidiaTeslaT4", "written_at": dt.datetime.utcnow().isoformat()}
    (SC / f"{system}.json").write_text(json.dumps(manifest, indent=2))
    print(f"SCORES[{system}]", sf.height, "rows", SC / f"{system}.parquet")
    joined = sf.join(lab, on=["imp_row", "cand_position"], how="left")
    assert joined["labels"].null_count() == 0
    per_imp = {k: np.asarray(v, float) for k, v in per_impression_metrics(impression_rows(joined, "score", label_col="labels")).items()}
    ci = bootstrap_evaluate(per_imp)
    ours = {k: float(np.nanmean(v)) for k, v in per_imp.items()}
    print(f"METRICS_OURS[{system}]", json.dumps(ours))
    print(f"METRICS_CI[{system}]", json.dumps({k: [c.mean, c.lo, c.hi] for k, c in ci.items()}))
    if not mask:
        pkg = cal_metric(labels, preds, ["group_auc", "mean_mrr", "ndcg@5;10"])
        print("METRICS_PKG", json.dumps(pkg))
        for k_ours, k_pkg in [("auc", "group_auc"), ("mrr", "mean_mrr"), ("ndcg@5", "ndcg@5"), ("ndcg@10", "ndcg@10")]:
            d = abs(round(ours[k_ours], 4) - pkg[k_pkg]); print(f"AGREE {k_ours} |Δ|={d:.2e}"); assert d <= 1e-4 + 1e-9
        print("METRICS_OURS", json.dumps(ours)); print("METRICS_CI", json.dumps({k: [c.mean, c.lo, c.hi] for k, c in ci.items()}))
    results[system] = (ours, pkg if not mask else None)
    stage(f"score_eval[{system}]")
sh(f"rm -rf {DATA} {WORK}/recommenders {WORK}/repo")   # keep only out/: `kaggle kernels output` pages at 20 files
print(f"RESULT PASS epochs={EPOCHS} group_auc={results['nrms_fresh'][1]['group_auc']} full_slate_auc={results['nrms_fresh'][0]['auc']:.6f} masked_auc={results['nrms_fresh_masked'][0]['auc']:.6f} total_seconds={time.time() - T0:.0f}")
