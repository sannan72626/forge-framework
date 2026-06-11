# FORGE Weekly Reports Index
## Fisher-Orchestrated Robust Gating for Erasure · EMNLP 2026

Each report covers: what was built, fine-tuning techniques, bugs fixed, results with metric definitions, and gating verdict.

| Week | Report | Status | Key Result |
|------|--------|--------|------------|
| Wk1–3 | [weeks1_3_report.md](weeks1_3_report.md) | ✅ PASS | 3 gaps identified; 7 loss functions; FORGEScheduler |
| Wk4 | [week4_report.md](week4_report.md) | ✅ PASS | FORGE FQ=0.869 (+0.473 vs LoKU); 8 bugs fixed |
| Wk5 | (included in Wk1-3 report) | ✅ PASS | rank=32 confirmed; lr=2e-5 corrected from simulation |
| Wk6 | [week6_report.md](week6_report.md) | ⚠️ SOFT WARN | ΔMU=0.061 (target <0.05); FORGE best scaling; 3 oracles built |
| Wk7 | [week7_report.md](week7_report.md) | ⚠️ PARTIAL | VerbMem PASS; WMDP/KnowMem WARN (MCQ format fix needed) |
| Wk8 | [week8_report.md](week8_report.md) | ✅ PASS | 54-run grid; FORGE FQ×MU=0.413 (4.4× next best); 4/5 sig p<0.05 |
| Wk9 | [week9_report.md](week9_report.md) | ✅ PASS | 43× speedup, 0.44% params; FORGE ΔFQ=0.048 (seq stability) |
| Wk10 | _pending_ | ⏳ | Paper writing (intro, related, method) |
| Wk11 | _pending_ | ⏳ | Results + analysis + compile PDF |
| Wk12 | _pending_ | ⏳ | Revisions + slides + demo |

## Report Template (for future weeks)
Each report must include:
1. **Goal** — what CLAUDE.md specifies for this week
2. **What we built** — scripts, classes, configs
3. **Fine-tuning techniques** — model, method, hyperparams, training config
4. **Bugs discovered and fixed** — symptom, cause, fix
5. **Results table** — all metrics with definitions
6. **Gating verdict** — pass/warn/fail with evidence
7. **Outputs produced** — file list
