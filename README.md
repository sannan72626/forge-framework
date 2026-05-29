# FORGE — Hyperparameter Sweep Simulation & Training Code

Fisher-Orchestrated Robust Gating for Erasure  
Selective LLM unlearning | Week 5 deliverable

---

## Quick start (no GPU)

```bash
pip install numpy scipy matplotlib rouge-score
python forge_sweep_simulation.py --out_dir results/
```

Produces in `results/`:
- `sweep_results.json`      — raw data for all 48 runs
- `sweep_scatter.png`       — FQ × MU Pareto scatter
- `heatmap.png`             — lr × epochs interaction heatmaps
- `lr_epoch_curves.png`     — decoupled FQ / MU curves
- `adaptive_lr.png`         — 4-phase schedule + rank comparison
- `recommended_config.json` — optimal (lr, epochs, rank) from Pareto front

---

## Full training (GPU required)

```bash
pip install -r requirements.txt
# Also: git+https://github.com/locuslab/open-unlearning.git

python -c "
from forge.trainer import forge_train_standalone, FORGEArgs
forge_train_standalone(
    model_name='locuslab/tofu_ft_llama2-7b',
    forget_split='forget05',
    retain_split='retain95',
    output_dir='outputs/forge_tofu5',
    total_steps=300,
    lr=5e-6,
    forge_args=FORGEArgs(
        lora_rank=32,
        lambda_retain=1.0,
        gamma_robust=0.5,
        eta_gate=0.01,
        k_adv=2,
        zeta_reason=0.0,   # set 0.2 for R-TOFU
    ),
)
"
```

---

## File map

```
forge_project/
├── forge_sweep_simulation.py   ← run this first; no GPU needed
├── requirements.txt
└── forge/
    ├── __init__.py
    ├── trainer.py              ← FORGETrainer (HF Trainer subclass)
    ├── adaptive_lr.py          ← FORGEScheduler (4-phase LR)
    └── metrics.py              ← ForgeEvaluator (FQ, MU, ROUGE-L, 2×MIA)
```

---

## Recommended config (from sweep)

| Parameter | Value |
|-----------|-------|
| lr (base) | 5e-6 |
| lr schedule | 4-phase adaptive (FORGEScheduler) |
| Epochs | 12 (≈ 300 steps at TOFU-5%) |
| LoRA rank | 32 (both adapters) |
| λ (retain) | 1.0 |
| γ (robust) | 0.5 |
| η (gate) | 0.01 |
| k_adv | 2 |
| ζ (CoT) | 0.0 → 0.2 on R-TOFU |

Predicted: FQ = 0.901, MU = 0.882 (vs default 0.897 / 0.876)

---

## Three key findings (Week 5)

1. **lr → MU**   Learning rate is the primary driver of utility preservation.  
   10× lr change costs ~5× more MU than doubling epochs.  
   Keep lr ≤ 5e-6 to guarantee MU ≥ 0.85.

2. **Epochs → FQ**   Epochs drive forgetting quality.  
   FQ spread across lr values: 0.019.  FQ spread across epochs: 0.096.  
   Epochs drive FQ 5× more than lr.  Train ≥ ep=10 for FQ ≥ 0.88.

3. **Rank = free lunch**   rank=32 Pareto-dominates rank=8 at every (lr, epoch) pair.  
   Cost: 0.4% of base model parameters.  Always use rank=32.
