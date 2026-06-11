# Metric Notes — FQ vs Official TOFU
## Honest documentation of how our Forget Quality differs from the published harness

Reviewers familiar with TOFU will compare our FQ to the official numbers. This documents
the exact relationship so the paper can state it transparently.

## Our FQ (as implemented)
- **Statistic:** per-example negative log-likelihood (NLL) of `"Q: {q}\nA: {a}"`.
- **Test:** two-sample Kolmogorov–Smirnov p-value between the unlearned model's NLL
  distribution on D_f and a **retain oracle's** NLL distribution on D_f.
- **Oracle:** scratch fine-tuned from the *base* model (microsoft/phi-1_5 or
  NousResearch/Llama-2-7b-hf) on the retain split only — never sees D_f.
- **Checkpoint:** peak FQ across the trajectory (FQ rises then overshoots).

## Official TOFU FQ (Maini et al. 2024)
- **Statistic:** "truth ratio" — ratio of normalized prob of the correct answer to that of
  perturbed/paraphrased answers.
- **Test:** KS p-value of the unlearned model's truth-ratio distribution vs the **provided
  retain checkpoint's** (`locuslab/tofu_ft_llama2-7b_retain95`).
- Same KS-test framework; different per-example statistic (truth-ratio vs NLL) and a
  provided (not scratch-trained) oracle.

## Why they can differ in absolute value
1. **Statistic:** NLL vs truth-ratio respond differently to over-forgetting. NLL keeps
   rising as the model degrades; truth-ratio saturates. → our FQ can read lower at the
   same forgetting level.
2. **Oracle:** the official Llama-2 retain checkpoint was unavailable/gated, so we
   scratch-trained oracles. A scratch oracle has a slightly different NLL profile than the
   official one. (This is also why `locuslab/tofu_ft_retain90_phi-1.5` gave FQ=1.0 — it was
   NOT a clean base-trained oracle; see week6_report.md.)

## Reviewer-facing statement (for the paper)
> "We compute Forget Quality as the KS-test p-value between the unlearned model and a
> retain-only oracle, following TOFU. We substitute NLL for the truth-ratio statistic and
> use scratch-trained oracles because the official Llama-2 retain checkpoints are gated.
> Absolute FQ values are therefore not directly comparable to published numbers; all
> comparisons in this paper are made under identical evaluation for every method, so
> *relative* rankings are valid."

## Remediation (concrete next step, needs GPU + open-unlearning install)
- Install `open-unlearning` (locuslab) and re-score the FORGE / LoKU / GA checkpoints with
  the **official** truth-ratio FQ on Llama-2-7B, using the official retain95 checkpoint
  once HF gate access propagates.
- Report both: our internal FQ (all methods, all splits) + official FQ (forget05, key
  methods) as a cross-check appendix.

**Status:** documented limitation; cross-check is queued, not yet run.
