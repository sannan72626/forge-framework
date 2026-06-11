# Project Status
current_week: 10
last_task: Wk5 sweep simulation complete
blocking: none

## Completed
- [x] Wk1: outputs/wk1/bibliography.md (10 papers)
- [x] Wk1: outputs/wk1/gap_analysis.md (3 gaps)
- [x] Wk1: outputs/wk1/gap_recommendation.md
- [x] Wk2: forge/metrics.py (ForgeEvaluator) — imports OK
- [x] Wk2: outputs/wk2/trajectory_mockup.png
- [x] Wk2: outputs/wk2/protocol.md
- [x] Wk3: forge/adaptive_lr.py (FORGEScheduler)
- [x] Wk3: forge/trainer.py (all 7 loss functions + METHOD_LOSSES)
- [x] Wk3: configs/forge_default.yaml (lr=5e-6, rank=32, ep=12)
- [x] Wk3: outputs/wk3/method_summary.md
- [x] Wk5: run forge_sweep_simulation.py → 48 runs
- [x] Wk5: outputs/wk5/recommended_config.json (rank=32 theoretical; rank=16 on 8 GB)
- [x] Wk5: outputs/wk5/heatmap.png + sweep_scatter.png + lr_epoch_curves.png + adaptive_lr.png

## Notes
- Hardware: RTX 3090 23 GB — use rank=32, batch=2, max_length=256, bfloat16 (no 4-bit needed)
- Target model: locuslab/tofu_ft_llama2-7b (pre-finetuned on full TOFU, no Task 0 needed)
- Reference model: locuslab/tofu_ft_llama2-7b_retain95 (for forget-5% FQ eval, provided by locuslab)
- Task 0 (Phi-3-mini) is for Wk6 cross-model generalization only, NOT a Wk4 blocker
- scripts/run_experiments.py is complete and GPU-ready

## TODO — Remaining tasks
- [x] Wk4: scripts/run_experiments.py (smoke + full modes) ← DONE
- [x] Wk4: smoke test PASS (GA + FORGE, 30 steps, RTX 3090)
- [x] Wk4: all 5 methods ran 300 steps — pipeline verified, MU/ROUGE/MIA valid
- [x] Wk4: COMPLETE — Phi-1.5 + TOFU forget05, valid oracle (retain95 from microsoft/phi-1_5)
- [x] Wk4: FORGE peak FQ=0.869 (step 125), MU=0.488 — +0.473 FQ advantage over LoKU (0.396)
- [x] Wk4: fig_trajectory.png saved → outputs/wk4/phi15/fig_trajectory.png
- [x] Wk4: l_rob bug fixed (was minimizing, now gradient ascent on token-dropped forget)
- [ ] Wk4: outputs/wk4/trajectories.json
- [ ] Wk4: outputs/wk4/fig_trajectory.png
- [x] Wk6: multi-split runs complete (forget01/05/10 × 4 methods)
- [x] Wk6: 3 oracles fine-tuned from base Phi-1.5 (retain99/95/90)
- [x] Wk6: comparison_table.csv saved; FORGE ΔMU=0.061 (target <0.05, soft WARN)
- [x] Wk6: FORGE has smallest ΔMU of any forgetting method; rank order preserved on forget05
- [ ] Wk6: cross-model runs (deferred to Wk8 — Phi-2 or OPT if time allows)
- [x] Wk7: COMPLETE — scripts/run_wmdp.py + scripts/run_muse.py + configs/forge_wmdp.yaml
- [x] Wk7: FORGE WMDP acc=0.370 (WARN — MCQ alignment gap; proper format needed in Wk8)
- [x] Wk7: FORGE MUSE VerbMem=0.000 PASS; KnowMem=0.115 WARN (best among retain-preserving)
- [x] Wk7: generalization_table.csv saved → outputs/wk7/
- [ ] Wk8: add GD+KL to forge/trainer.py (already done in METHOD_LOSSES)
- [ ] Wk8: forge/stats.py
- [ ] Wk8: 63-run grid complete
- [ ] Wk8: outputs/wk8/stats_report.json (all p<0.05)
- [ ] Wk9: scripts/run_sequential.py
- [ ] Wk9: scripts/cost_analysis.py
- [ ] Wk9: outputs/wk9/sequential_FORGE/FORGE_sequential.json
- [ ] Wk10: paper/sections/intro.tex (650-800w)
- [ ] Wk10: paper/sections/related.tex (900-1100w)
- [ ] Wk10: paper/sections/method.tex (1400-1600w)
- [ ] Wk11: paper/sections/results.tex + analysis.tex + abstract.tex
- [ ] Wk11: paper/main.pdf compiled (<=9 pages)
- [ ] Wk12: outputs/presentation/FORGE_slides.pptx (15 slides)
- [ ] Wk12: demo/forge_demo_executed.ipynb
- [ ] Wk12: forge_release.zip (submission ready)
