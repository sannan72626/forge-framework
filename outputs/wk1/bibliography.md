# Annotated Bibliography — LLM Unlearning

## Core Benchmarks

**[1] TOFU** Maini et al. (ICLR 2024). "TOFU: A Task of Fictitious Unlearning for LLMs."
- Benchmark: fictional author profiles (200 authors × 20 QA pairs each)
- Metrics: Forget Quality (KS-test p-value), Model Utility (retain-set performance)
- Key finding: most unlearning methods fail at least one of FQ or MU
- Gap identified: no method simultaneously achieves high FQ AND high MU

**[2] WMDP** Li et al. (2024). "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning."
- Benchmark: 4-choice MCQ on biosecurity / chemical / cyber hazards
- Target: accuracy → 25% (random chance) while preserving MMLU capability
- Gap: no method designed for MCQ-format forgetting with capability preservation

**[3] MUSE** Shi et al. (ICLR 2025). "MUSE: Machine Unlearning Six-Way Evaluation."
- Benchmark: copyright news + book corpora; verbatim memorisation metrics
- Metrics: VerbMem (n-gram overlap), KnowMem (ROUGE-L on factual QA)
- Gap: no benchmark tests both verbatim AND knowledge forgetting simultaneously

## Unlearning Methods

**[4] FLAT** Wang et al. (ICLR 2025). "FLAT: Federated Learning-inspired Asymmetric Token Optimization."
- Method: forget-only loss adjustment; no retain set required
- Limitation: over-suppresses retain-set performance without explicit retain loss
- Gap B exposed: no adversarial robustness training

**[5] LoKU = IHL+FILA** Cha et al. (ICLR 2025). "Towards Robust Unlearning with IHL and FILA."
- Method: Inverted Hinge Loss (IHL) + Fisher Initialised LoRA Adapter (FILA)
- Best published PEFT baseline; FQ=0.781, MU=0.853 @ TOFU-5%
- Limitation: single adapter (no separation of forget/retain); no adversarial training
- Gap A: dual adapters would decouple forget/retain gradients

**[6] NPO** Zhang et al. (2024). "Negative Preference Optimization."
- Method: preference-based; treats forget examples as "negative" and optimises away
- Limitation: sensitive to beta hyperparameter; MU degrades with larger forget sets

**[7] Closer Look** Yuan et al. (ICLR 2025). "A Closer Look at Machine Unlearning for LLMs."
- Comprehensive evaluation of GA, GD, KL, NPO, SCRUB on TOFU splits
- Shows MU collapses dramatically at forget-10% for most methods
- Gap: no method tested for graceful scaling behaviour

## Critical Perspectives

**[8] CMU critique** Thaker et al. (SaTML 2025). "Position: LLM Unlearning Benchmarks are Weak Proxies."
- Shows answer-level FQ can converge while reasoning traces still leak
- Motivates Gap C (CoT-aware unlearning)
- Key experiment: CoT-prefix prompts recover "forgotten" content

**[9] R-TOFU** Yoon et al. (EMNLP 2025). "R-TOFU: Reasoning-aware TOFU for LLMs with Chain-of-Thought."
- Extends TOFU with reasoning trace evaluation; introduces step-wise KL metric
- Target: per-step CoT distribution pushed away from pre-unlearn reference
- Gap C: step-wise ζ·ℒ_reason needed for reasoning-model safety

## Surveys

**[10] Survey** Nguyen et al. (ACM TIST 2025). "A Survey of Machine Unlearning for Large Language Models."
- Taxonomy: exact vs approximate, centralised vs federated, benchmark vs deployment
- Key insight: deployment requires retain-free and modular designs
- Gap: no existing method is simultaneously selective, robust, AND modular
