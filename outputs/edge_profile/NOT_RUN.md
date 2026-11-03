# Edge/IoT Deployment Profiling — NOT RUN

**Status: not executed. No edge or Jetson-class hardware is available in
this environment.**

## Hardware actually available (verified, `docs/experiment_audit.md`)

```
NVIDIA GeForce RTX 3090, 24576 MiB
platform.machine(): x86_64
```

This is a workstation/desktop GPU, not an edge device. `scripts/profile_edge_unlearning.py`
includes an explicit guard (`is_edge_device` check on `platform.machine()`
and device name) that refuses to silently label an RTX 3090 run as edge
evidence — it would print a warning and set `is_edge_device: false` /
`warning: "...do not cite these numbers as edge/IoT deployment evidence"`
rather than fabricate a result.

## What this means for the manuscript

Per the Phase 7 acceptance criteria: **the abstract and conclusion must
not claim edge/IoT deployment has been proven or validated.** The existing
cost table (Table IX: 6.29M trainable params, 4.61 GB peak VRAM, 43.0×
speedup over full retrain) is real, but it is a **workstation-GPU
resource-efficiency result**, not an edge-deployment result. Recommended
manuscript language: replace "supporting practical unlearning for
privacy-preserving IoT model maintenance" / similar deployment-proof
claims with "resource-efficient relative to full retraining on a
workstation GPU; edge-hardware validation (e.g. Jetson Orin, which the
4.61 GB peak VRAM figure suggests may be feasible) is left to future work."
This matches reviewer findings [R2] and [R19].

## If edge hardware becomes available later

Run:
```bash
python scripts/profile_edge_unlearning.py \
  --model locuslab/tofu_ft_phi-1.5 --base_model microsoft/phi-1_5 \
  --method FORGE --split forget05 --steps 25 --batch_size 1 \
  --output outputs/edge_profile/forge_phi15_edge.json
```
The script measures peak memory, seconds/step, total time, and attempts an
energy reading via `tegrastats` (Jetson) or `nvidia-smi power.draw` as a
fallback, returning `null` (never a fabricated number) if neither tool is
present. It will refuse to mark output as edge evidence unless run on
`aarch64`/a device name containing `jetson`/`orin`/`xavier`/`nano`.
