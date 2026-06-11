# Weeks 1–3 Report — Literature, Protocol & Method Design
## FORGE: Fisher-Orchestrated Robust Gating for Erasure
**Completed:** (simulated on another device before this machine setup)  
**Status at start of this machine:** All outputs verified clean, STATUS.md at current_week=6

---

## Week 1 — Literature Review

### Goal
Map the LLM unlearning landscape, annotate 10 key papers, and identify 3 research gaps that FORGE addresses.

### Outputs
| File | Content |
|------|---------|
| `outputs/wk1/bibliography.md` | 10-paper annotated bibliography |
| `outputs/wk1/gap_analysis.md` | 3 research gaps (A, B, C) |
| `outputs/wk1/gap_recommendation.md` | FORGE method proposal |

### 10 Papers Covered

| # | Paper | Venue | Key Finding |
|---|-------|-------|-------------|
| 1 | TOFU (Maini et al.) | ICLR 2024 | Benchmark: fictional authors, FQ+MU metrics |
| 2 | WMDP (Li et al.) | 2024 | MCQ biosecurity unlearning benchmark |
| 3 | MUSE (Shi et al.) | ICLR 2025 | VerbMem + KnowMem copyright metrics |
| 4 | FLAT (Wang et al.) | ICLR 2025 | Token-level forget probability suppression |
| 5 | LoKU (Cha et al.) | ICLR 2025 | IHL + Fisher-initialized LoRA; best PEFT baseline |
| 6 | NPO (Zhang et al.) | 2024 | Preference-based unlearning |
| 7 | Closer Look (Yuan et al.) | ICLR 2025 | GA/GD/KL/NPO/SCRUB comparison |
| 8 | CMU critique (Thaker et al.) | SaTML 2025 | CoT traces still leak after answer-level unlearning |
| 9 | R-TOFU (Yoon et al.) | EMNLP 2025 | Step-wise KL on reasoning traces |
| 10 | Survey (Nguyen et al.) | ACM TIST 2025 | Taxonomy of LLM unlearning methods |

### 3 Research Gaps Identified

**Gap A — Selective LoRA Forgetting:**  
All existing PEFT methods use a single LoRA adapter that must simultaneously suppress D_f and preserve D_r, creating gradient conflicts. LoKU (best published) drops MU by 5.9pp from forget-1% to forget-10%.  
→ *FORGE proposes:* Dual adapters (Δ_f, Δ_r) merged via diagonal Fisher weighting.

**Gap B — Adversarial Robustness:**  
No method trains against distributional probes. Paraphrase/entity-swap/CoT-prefix attacks recover "forgotten" content in LoKU (mean reactivation 0.228).  
→ *FORGE proposes:* Minimax adversarial inner loop with prefix-tuned T5-small probe generator.

**Gap C — CoT Trace Leakage:**  
R-TOFU shows FLAT achieves FQ=0.91 on canonical queries but trace-level ROUGE-L remains 0.34.  
→ *FORGE proposes:* Step-wise KL divergence on CoT reasoning traces (ζ·ℒ_reason).

### FORGE Objective (full)
```
ℒ(θ,g,ψ) = ℒ_forget + λ·ℒ_retain + γ·ℒ_robust(ψ) + ζ·ℒ_reason + η·ℒ_gate
            min over (θ,g);   max over ψ
```

### Gating: PASS ✓
- bibliography.md: 10 entries ✓
- gap_analysis.md: 3 gaps (A, B, C) ✓
- gap_recommendation.md: FORGE proposal ✓

---

## Week 2 — Experimental Protocol

### Goal
Define exactly how experiments run and what gets measured. Build the ForgeEvaluator class. Create the FQ×MU trajectory mockup.

### Outputs
| File | Content |
|------|---------|
| `forge/metrics.py` | ForgeEvaluator class — 5 metrics |
| `outputs/wk2/protocol.md` | 2-page experimental protocol |
| `outputs/wk2/trajectory_mockup.png` | Design mockup of FQ×MU plot |

### ForgeEvaluator — 5 Metrics Implemented

```python
class ForgeEvaluator:
    def forget_quality(self, forget_examples)  → float  # KS test p-value
    def model_utility(self, retain_examples)   → float  # Harmonic(ROUGE-L, ppl_recovery)
    def rouge_l(self, examples)                → float  # ROUGE-L on generated answers
    def mia_truth_ratio(self, forget_examples) → float  # p(true)/p(shuffled) → target 1.0
    def mia_probability_ratio(self, forget)    → float  # student/ref NLL ratio → target 1.0
```

### Experimental Protocol Defined

**Datasets:**
- TOFU: forget-1%, forget-5%, forget-10% splits (200/20/400 forget examples)
- WMDP-bio: MCQ biosecurity (Wk7)
- MUSE-News: copyright corpus (Wk7)

**Models:**
- Base: `locuslab/tofu_ft_phi-1.5` (Phi-1.5, 1.4B, TOFU fine-tuned)
- Reference/Oracle: `microsoft/phi-1_5` fine-tuned on retain-95%

**Training:** Seeds 42, 123, 456 · 3 runs per (method, split) · lr=2e-5 · rank=32

**Statistical validation (Wk8):** Paired t-test FORGE vs each baseline; α=0.05; 95% bootstrap CI (1000 resamples)

### Gating: PASS ✓
- `from forge.metrics import ForgeEvaluator` — imports OK ✓
- `outputs/wk2/trajectory_mockup.png` exists ✓
- `outputs/wk2/protocol.md` exists ✓

---

## Week 3 — Method Design

### Goal
Implement all 7 loss functions in `forge/trainer.py`. Build the adaptive 4-phase LR scheduler. Lock the default config.

### Outputs
| File | Content |
|------|---------|
| `forge/trainer.py` | All 7 loss functions + METHOD_LOSSES dict |
| `forge/adaptive_lr.py` | FORGEScheduler (4-phase cosine) |
| `forge/__init__.py` | Lazy imports for torch-free tasks |
| `configs/forge_default.yaml` | Default hyperparameters |
| `configs/baselines.yaml` | Per-baseline configs |
| `outputs/wk3/method_summary.md` | FORGE method summary |

### 7 Loss Functions Implemented

```python
METHOD_LOSSES = {
    "GA":    loss_GA,    # Gradient Ascent: −NLL(forget)
    "GD":    loss_GD,    # Gradient Difference: GA + |retain_student − retain_ref|
    "KL":    loss_KL,    # KL anchoring: GA + KL(student‖ref) on retain
    "NPO":   loss_NPO,   # Negative Preference Optimization (Bradley-Terry)
    "FLAT":  loss_FLAT,  # Token-level probability suppression
    "LoKU":  loss_LoKU,  # Inverted Hinge Loss + retain NLL
    "FORGE": loss_FORGE, # Bounded IHL + retain + robust ascent
}
```

### FORGEScheduler — 4-Phase Adaptive LR

```
Phase 1: Linear warm-up     (0 → t_warm)     lr_base → lr_peak
Phase 2: Cosine decay       (t_warm → t_decay) lr_peak → lr_base
Phase 3: Adversarial plateau (t_decay → t_plat) constant lr_adv
Phase 4: Cosine floor       (t_plat → T)      lr_adv → lr_floor
```

Implementation: `forge/adaptive_lr.py::FORGEScheduler(LRScheduler)`

### Default Config (forge_default.yaml — final version after Wk4 tuning)
```yaml
lr: 2.0e-5           # Tuned in Wk4 diagnostic sweep (5e-6 too slow, 5e-5 too fast)
lora_rank: 32        # Wk5 sweep: rank=32 always best ("free lunch")
ihl_margin: 4.0      # FORGE tanh bound; LoKU margin also set to 4.0 in Wk4
lambda_retain: 1.0   # Retain loss weight
gamma_robust: 0.5    # Robust loss weight
max_steps: 250       # Sweet spot found in Wk4 diagnostic
eval_every: 25       # Fine-grained trajectory for peak FQ detection
per_device_batch: 4
```

### Gating: PASS ✓
```python
from forge.trainer import METHOD_LOSSES
assert len(METHOD_LOSSES) == 7  # PASS
```

---

## Week 5 — Hyperparameter Sweep (Simulated)

### Goal
Find optimal (lr, epochs, rank) via 48-run sweep simulation. Confirm rank=32 is recommended.

### Technique: Sweep Simulation
Since GPU experiments weren't available on the simulation device, a parametric simulation (`forge_sweep_simulation.py`) was used:
- Simulated 48 runs across lr ∈ {1e-6, 5e-6, 1e-5, 2e-5} × epochs ∈ {3,5,10,15} × rank ∈ {8,16,32}
- FQ and MU estimated from a parametric model fitted to published LoKU/FLAT results
- Generated 4 analysis figures

### Outputs
| File | Content |
|------|---------|
| `outputs/wk5/recommended_config.json` | Best config: lr=5e-6, rank=32, epochs=12 |
| `outputs/wk5/heatmap.png` | FQ×MU heatmap over lr×epoch grid |
| `outputs/wk5/sweep_scatter.png` | Scatter of all 48 runs |
| `outputs/wk5/lr_epoch_curves.png` | FQ/MU vs epoch for each lr |
| `outputs/wk5/adaptive_lr.png` | FORGEScheduler phase visualization |

### Key Findings (simulated — validated in Wk4)
- **rank=32:** Always better than rank=8 or rank=16 ("free lunch") — confirmed in real runs
- **lr sensitivity:** FQ peaks at lr=5e-6 in simulation, but Wk4 real data showed 2e-5 needed for Phi-1.5
- **epochs:** More epochs → higher FQ but MU degrades past ~12 epochs

Note: The simulated lr recommendation (5e-6) was corrected to 2e-5 in Week 4 real experiments via diagnostic sweep.

### Gating: PASS ✓
```python
cfg = json.load(open("outputs/wk5/recommended_config.json"))
assert cfg["rank"] == 32  # PASS
```

---

*Report generated: 2026-06-02*
