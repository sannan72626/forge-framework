# FORGE Research Roadmap — Weeks 1–12
## Target: EMNLP 2026 · Fisher-Orchestrated Robust Gating for Erasure

---

## At-a-glance

```
Phase 1 │ Phase 2 │ Phase 3   │ Phase 4      │ Phase 5     │ Phase 6  │ Phase 7
Lit+Proto│ Method  │ Exp (1)   │ Generalize   │ Analysis    │ Writing  │ Revise
Wk 1-2  │ Wk 3    │ Wk 4-5-6  │ Wk 7-8       │ Wk 9        │ Wk 10-11 │ Wk 12
         │         │           │              │             │          │
         │         │ ◆ M1 (Wk4)│ ◆ M2 (Wk8)  │◆ Cost (Wk9) │◆ Draft   │◆ Submit
```

---

## Week-by-week

### ✅ Week 1 — Literature Review  `[COMPLETE]`
**Goal:** Map the unlearning landscape; identify 3 research gaps.

| Task | Output |
|------|--------|
| Read & annotate 10 papers | `outputs/wk1/bibliography.md` |
| Identify gaps A/B/C | `outputs/wk1/gap_analysis.md` |
| Write 1-page gap recommendation | `outputs/wk1/gap_recommendation.pdf` |

**Papers:** TOFU (Maini 2024), WMDP, MUSE, FLAT (ICLR'25), LoKU (ICLR'25),
Closer Look (ICLR'25), CMU critique, R-TOFU (EMNLP'25), ACM TIST survey, WWW'26

**Gaps identified:**
- **A** — Selective forgetting: LoRA-based, Fisher-weighted, token-gated
- **B** — Adversarial robustness: paraphrase / decode-shift attacks
- **C** — CoT trace leakage: step-wise KL on reasoning traces

---

### ✅ Week 2 — Experimental Protocol  `[COMPLETE]`
**Goal:** Define exactly how experiments will be run and evaluated.

| Task | Output |
|------|--------|
| Define 3 forget-set splits | TOFU 1%, 5%, 10% |
| Define 5 evaluation metrics | FQ, MU, ROUGE-L (f/r), MIA-TR, MIA-PR |
| Design FQ×MU trajectory frame | `outputs/wk2/trajectory_mockup.png` |
| Write 2-page protocol doc | `outputs/wk2/protocol.docx` |

---

### ✅ Week 3 — FORGE Method Design  `[COMPLETE]`
**Goal:** Full formal specification of the novel method.

| Task | Output |
|------|--------|
| Gap A: Fisher-weighted dual adapter + token gate | `forge/trainer.py` |
| Gap B: Adversarial inner loop Φ_ψ | `forge/trainer.py` |
| Gap C: Step-wise CoT KL loss | `forge/trainer.py` |
| Pseudocode + ablation plan | `outputs/wk3/method_section.docx` |

**Full objective:**
```
ℒ(θ,g,ψ) = ℒ_forget + λ·ℒ_retain + γ·ℒ_robust(ψ) + ζ·ℒ_reason + η·ℒ_gate
           min over (θ,g);   max over ψ
```

---

### ✅ Week 4 — Initial Experiments (M1 Report)  `[COMPLETE]`
**Goal:** Produce first FQ×MU trajectory; adversarial robustness audit.

| Task | Output |
|------|--------|
| TOFU-5% runs: GA, NPO, FLAT, LoKU, FORGE | `outputs/wk4/trajectories.json` |
| 5-template adversarial robustness | `outputs/wk4/fig_robustness.png` |
| M1 Report | `outputs/wk4/M1_Report_FORGE.docx` |

**Key results (illustrative):** FORGE FQ=0.897, MU=0.876 · adv.react=0.070 vs LoKU 0.228

**⚠️ NEEDS REAL DATA:** Replace illustrative results with `run_experiments.py --mode full`

---

### ✅ Week 5 — Hyperparameter Sweep  `[COMPLETE]`
**Goal:** Find optimal (lr, epochs, rank) for FORGE.

| Task | Output |
|------|--------|
| 48-run sweep (4×4×3) | `outputs/wk5/sweep_results.json` |
| Heatmap analysis | `outputs/wk5/heatmap.png` |
| Adaptive 4-phase LR design | `forge/adaptive_lr.py` |
| Recommended config | `configs/forge_default.yaml` |

**Findings:** lr→MU, epochs→FQ, rank=free lunch (rank=32 always best).
**Recommended:** lr=5e-6, adaptive, ep=12, rank=32.

---

### ✅ Week 6 — Multi-split + Cross-model  `[COMPLETE]`
**Goal:** Validate scaling and transfer.

| Task | Output |
|------|--------|
| Multi-split (1%/5%/10%) full comparison | `outputs/wk6/comparison_table.csv` |
| Cross-model: Phi-1.5B, OPT-2.7B | `outputs/wk6/crossmodel_results.json` |
| Degradation index analysis | `outputs/wk6/w6_degradation_index.png` |
| M2 Report | `outputs/wk6/M2_Report_FORGE.docx` |

**Key finding:** FORGE ΔMU=0.021 (1%→10%) vs GA ΔMU=0.469. Rank order preserved across all model sizes.

---

### 🔲 Week 7 — Benchmark Generalization: WMDP + MUSE  `[TODO]`
**Dates:** May 25 – Jun 1

| # | Task | Command | Output |
|---|------|---------|--------|
| 7.1 | Adapt FORGE to WMDP-bio (MCQ) | `python scripts/run_wmdp.py` | `outputs/wk7/wmdp_results.json` |
| 7.2 | Adapt FORGE to MUSE-News | `python scripts/run_muse.py` | `outputs/wk7/muse_results.json` |
| 7.3 | Run 5 baselines on WMDP + MUSE | (loop in CLAUDE.md §7) | `outputs/wk7/*/` |
| 7.4 | Generalization table | `plot_all.py --task generalization` | `generalization_table.csv` |

**WMDP target:** MCQ accuracy ≤ 0.27 (random=0.25) on bio forget set
**MUSE target:** VerbMem ≤ 0.05, KnowMem ≤ 0.05
**Retain target:** MMLU accuracy ≥ 0.58; MUSE retain perplexity within 10%

**New code needed:**
- `forge/wmdp_eval.py` — MCQ log-prob evaluation
- `forge/muse_eval.py` — VerbMem (n-gram overlap) + KnowMem (ROUGE-L)
- `configs/forge_wmdp.yaml` — higher lambda_retain=2.0; retain_dataset=mmlu

---

### 🔲 Week 8 — Full Experiments + Statistics  `[TODO]`
**Dates:** Jun 1 – Jun 8

| # | Task | Command | Output |
|---|------|---------|--------|
| 8.1 | Add GD + KL losses to trainer | Edit `forge/trainer.py` | — |
| 8.2 | Full grid: 7 methods × 3 splits × 3 seeds | `run_experiments.py` (63 runs) | `outputs/wk8/*/` |
| 8.3 | Paired t-tests (p<0.05) | `forge/stats.py` | `stats_report.json` |
| 8.4 | 95% bootstrap CIs (1000 resamples) | `forge/stats.py` | merged into stats_report |
| 8.5 | Main results table (LaTeX) | `plot_all.py --task paper_tables` | `paper/tables/main_table.tex` |
| 8.6 | Cross-benchmark results section draft | write `paper/sections/results_crossbench.tex` | — |

**GPU estimate:** 63 runs × ~1h each = 63 GPU-hours (parallelise with Slurm or multiple GPUs)

**Statistical requirements:**
```python
# Required: for each metric at forget-5%
FORGE vs GA:   p < 0.010
FORGE vs NPO:  p < 0.050
FORGE vs FLAT: p < 0.050
FORGE vs LoKU: p < 0.050
FORGE vs GD:   p < 0.050
FORGE vs KL:   p < 0.050
# If any p > 0.05, report effect size (Cohen's d) and note in limitations
```

**Milestone ◆ M2 Stats:** All tables with CIs, all p-values, draft results section.

---

### 🔲 Week 9 — Sequential Unlearning + Cost Analysis  `[TODO]`
**Dates:** Jun 8 – Jun 15

| # | Task | Command | Output |
|---|------|---------|--------|
| 9.1 | Sequential: forget-1% → forget-5% on same model | `scripts/run_sequential.py` | `sequential_results.json` |
| 9.2 | Compare GA and LoKU on sequential | (same script, --method flag) | merged |
| 9.3 | Measure FLOPs, wall-clock, VRAM | `scripts/cost_analysis.py` | `cost_analysis.json` |
| 9.4 | vs full-retrain cost estimate | (in cost_analysis.py) | `cost_table.csv` |
| 9.5 | Cost comparison figure | `plot_all.py --task cost` | `cost_comparison.png` |

**Sequential analysis key question:**
Does stage-2 unlearning degrade the stage-1 forget quality?
Target: FORGE ΔFQ(stage-1) < 0.05 after stage-2 training.

**Cost targets:**
- FORGE FLOPs: ≤ 5% of full retraining
- Peak VRAM: ≤ 24 GB (fits single consumer GPU)
- Wall-clock: report in minutes; compare to 10s of GPU-hours for retrain

**Milestone ◆ Cost Analysis:** Sequential table + cost table in paper.

---

### 🔲 Week 10 — Write Introduction, Related Work, Methodology  `[TODO]`
**Dates:** Jun 15 – Jun 22

| # | Section | Length | Key content |
|---|---------|--------|-------------|
| 10.1 | Introduction | 1.5 pp | Hook, problem, 3 contributions, outline |
| 10.2 | Related Work | 2.0 pp | Benchmarks, methods, PEFT editing, adversarial |
| 10.3 | Methodology | 3.0 pp | Notation, Gap A (Fisher gate), Gap B (adv. loop), Gap C (CoT KL), Algorithm 1 |

**Writing rules:**
- Every numeric claim → cite specific output file and experiment
- Algorithm 1 pseudocode must match `forge/trainer.py compute_loss()` exactly
- Figure 1 (architecture) = `outputs/wk3/fig_architecture.png`
- No passive voice in contributions ("We propose", not "A method is proposed")
- Page budget: EMNLP 2026 = 9 pages + unlimited references

---

### 🔲 Week 11 — Write Results + Analysis; Internal Review  `[TODO]`
**Dates:** Jun 22 – Jun 29

| # | Section | Length | Key content |
|---|---------|--------|-------------|
| 11.1 | Experiments (setup) | 0.5 pp | Models, datasets, baselines, metrics, seeds |
| 11.2 | Results (main table + figures) | 1.5 pp | Table 1 (7 methods × 5 metrics), Figure 2 (trajectory), Figure 3 (degradation) |
| 11.3 | Cross-benchmark results | 0.5 pp | Table 3 (WMDP + MUSE) |
| 11.4 | Analysis + Ablation | 1.5 pp | Ablation table, GA diagnosis, sweep insight |
| 11.5 | Conclusion + Abstract | 0.5 pp | 3 claims, limitations, future work |

**Ablation runs** (if not done in Wk 7):
```bash
for ZEROED in gamma lambda eta k_adv rank_8; do
  python scripts/run_experiments.py --mode ablation --ablate $ZEROED \
    --split forget05 --seed 42 --out_dir outputs/wk11/ablation_$ZEROED/
done
```

**Internal review:** Send PDF to Professor Kim + Angelo by **Jun 25**.
Response expected by Jun 28.

**Milestone ◆ Draft:** Complete paper PDF, all sections, all figures, all tables.

---

### 🔲 Week 12 — Revise + 15-min Presentation + Demo  `[TODO]`
**Dates:** Jun 29 – Jul 6

| # | Task | Output |
|---|------|--------|
| 12.1 | Incorporate Professor Kim + Angelo feedback | `paper/revision_notes.md` |
| 12.2 | Re-run any experiments requested in review | `outputs/wk12/*/` |
| 12.3 | Build 15-slide presentation | `outputs/presentation/FORGE_slides.pptx` |
| 12.4 | Build demo notebook (runs in Colab) | `demo/forge_demo.ipynb` |
| 12.5 | Final compile + anonymization check | `paper/main.pdf` |
| 12.6 | Prepare code release zip | `forge_release.zip` |

**Presentation structure (15 slides, 1 min/slide):**
1. Title · 2. Motivation · 3. Problem · 4. Prior work gap ·
5. FORGE architecture · 6. Algorithm · 7. Main results ·
8. Multi-split · 9. Cross-model · 10. WMDP+MUSE ·
11. Ablation · 12. Cost · 13. GA diagnosis ·
14. Limitations · 15. Conclusion

**Demo must run in <5 min, show:** forgetting (forget query ≠ correct answer) + utility (retain query = correct answer) + robustness (paraphrase probe → still forgotten).

**Milestone ◆ Submission:** EMNLP 2026 deadline (check ACL 2026 calendar for exact date).

---

## Dependency graph

```
Wk1 ──► Wk2 ──► Wk3 ──► Wk4 ──► Wk5 ──► Wk6
                   │               │
                   │               ├──► Wk7 ──► Wk8 ──► Wk9
                   │               │             │
                   └───────────────┘             └──► Wk10 ──► Wk11 ──► Wk12
```

Wk8 depends on: Wk4 (TOFU results), Wk7 (WMDP/MUSE), Wk5 (recommended config)
Wk10 depends on: all experiment outputs (Wk4-Wk9)
Wk11 ablations can run in parallel with Wk10 writing

---

## Resource budget

| Resource | Estimate |
|----------|---------|
| GPU-hours (training) | ~150 GPU-hours total (63 wk8 + ~40 wk7 + ~15 wk9 + misc) |
| GPU-hours (eval) | ~30 GPU-hours |
| Disk space | ~200GB (model checkpoints; compress after each week) |
| Minimum GPU | 24GB VRAM (A10, RTX 4090, or A100) for Llama-2-7B + LoRA |
| Recommended | 4× A100 80GB for parallelism (wk8 grid takes ~16h on 4 GPUs) |

**Checkpoint retention policy:** Keep only best seed per (method, split); delete others after stats are computed.

---

## Risk register

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| GPU access delay | Medium | Start TOFU smoke test (GPT-2) immediately; use cloud (Lambda/Vast.ai) if needed |
| WMDP adaptation harder than expected | Medium | Allow 2 days buffer in Wk7; adapt loss function if MCQ accuracy not dropping |
| p-values not significant (Wk8) | Low | Run 5 seeds instead of 3; report effect size alongside |
| Paper over page limit | Medium | Cut cost analysis to supplementary; condense cross-model to paragraph |
| Review feedback requires new experiment | Medium | Budget Wk12 Days 1-4 for new runs; days 5-7 for revision only |

---

## Contacts

| Role | Person | Availability |
|------|--------|-------------|
| Supervisor | Professor Kim | Mon 10:00 weekly check-in |
| Collaborator | Angelo | MUSE experiments (Wk7) |
| GPU cluster admin | — | Slack for OOM/quota issues |
