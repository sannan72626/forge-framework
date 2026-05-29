# Project Status
current_week: 6
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
- [x] Wk3: configs/forge_default.yaml (lr=5e-6, rank=16, ep=10)
- [x] Wk3: outputs/wk3/method_summary.md
- [x] Wk5: run forge_sweep_simulation.py → 48 runs
- [x] Wk5: outputs/wk5/recommended_config.json (rank=32 theoretical; rank=16 on 8 GB)
- [x] Wk5: outputs/wk5/heatmap.png + sweep_scatter.png + lr_epoch_curves.png + adaptive_lr.png

## Notes
- Hardware: RTX 3050 8 GB — all configs use rank=16, batch=1, max_length=128, 4-bit quant
- Target model: microsoft/Phi-3-mini-4k-instruct (fine-tuned on TOFU, Task 0)
- Task 0 (Phi-3-mini TOFU fine-tune) must be done before Wk4 GPU experiments
- Wk4: requires GPU + Task 0 complete + scripts/run_experiments.py

## TODO — Remaining tasks
- [ ] Wk0: fine-tune Phi-3-mini on TOFU full → outputs/wk0/phi3_tofu_full/
- [ ] Wk4: scripts/run_experiments.py (smoke + full modes)
- [ ] Wk4: run smoke test (GPT-2, 30 steps)
- [ ] Wk4: outputs/wk4/trajectories.json
- [ ] Wk4: outputs/wk4/fig_trajectory.png
- [ ] Wk6: multi-split runs (1%/5%/10%)
- [ ] Wk6: cross-model runs (Phi/OPT/Llama)
- [ ] Wk6: outputs/wk6/comparison_table.csv
- [ ] Wk7: forge/wmdp_eval.py
- [ ] Wk7: forge/muse_eval.py
- [ ] Wk7: outputs/wk7/wmdp_summary.json
- [ ] Wk7: outputs/wk7/muse_summary.json
- [ ] Wk7: outputs/wk7/generalization_table.csv
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
