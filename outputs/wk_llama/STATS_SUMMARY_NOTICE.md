# Notice: `stats_summary.json` is superseded

`outputs/wk_llama/stats_summary.json` (public since the initial 2026-06-11
release) computes, for each method, `fq_mean` from each seed's **peak** FQ
step and `mu_mean` from each seed's **final** step — two different
checkpoints per seed, not the single checkpoint either value actually
came from. This produces misleading numbers: for example GA's `mu_mean`
in that file is `0.0`, because GA's peak-FQ step and final step are not
the same step, and GA's utility at the final step collapses to zero.

`outputs/wk_llama/stats_summary_corrected.json` fixes this by applying the
single-checkpoint-consistent selection rule used elsewhere in this repo
(`argmax_t(FQ_NLL_t * MU_ret_t)` per seed, `forge/stats.py::_select_checkpoint`)
and reporting `fq_mean`, `mu_mean`, and `j_mean` from the *same* selected
step. Under this rule, GA's `mu_mean` is ≈0.642, not `0.0`.

**Use `stats_summary_corrected.json`. Do not use `stats_summary.json` for
any reported number.**

The original file is left in place, unmodified — this notice does not
alter or delete historical data, it only redirects which file should be
read.
