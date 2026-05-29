"""
forge/metrics.py
────────────────
All five evaluation metrics for the FORGE protocol.

FQ    — Forget Quality (KS-test p-value against retain-only reference)
MU    — Model Utility (composite retain-set score)
RL    — ROUGE-L on forget / retain sets
MIA-TR — Membership Inference: Truth Ratio
MIA-PR — Membership Inference: Probability Ratio

Usage:
    from forge.metrics import ForgeEvaluator

    evaluator = ForgeEvaluator(model, ref_model, tokenizer, device)
    results = evaluator.evaluate(forget_examples, retain_examples)
    # returns dict: fq, mu, rouge_forget, rouge_retain, mia_tr, mia_pr
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    import types, functools

    class _NoGrad:
        """No-op stand-in for torch.no_grad() usable as decorator or context manager."""
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __call__(self, fn):
            @functools.wraps(fn)
            def wrapper(*a, **kw): return fn(*a, **kw)
            return wrapper

    torch = types.SimpleNamespace(
        no_grad=_NoGrad,
        device=lambda x: x,
    )
from scipy import stats
from rouge_score import rouge_scorer


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QAExample:
    question: str
    answer:   str
    is_forget: bool = False


@dataclass
class MetricResult:
    fq:           float   # Forget Quality (KS p-value)
    mu:           float   # Model Utility (composite)
    rouge_forget: float   # ROUGE-L on forget queries
    rouge_retain: float   # ROUGE-L on retain queries
    mia_tr:       float   # MIA Truth Ratio
    mia_pr:       float   # MIA Probability Ratio

    def __str__(self) -> str:
        return (
            f"FQ={self.fq:.3f}  MU={self.mu:.3f}  "
            f"RL-f={self.rouge_forget:.3f}  RL-r={self.rouge_retain:.3f}  "
            f"MIA-TR={self.mia_tr:.3f}  MIA-PR={self.mia_pr:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────
class ForgeEvaluator:
    """
    Evaluates the full 5-metric stack at a single model checkpoint.

    Args:
        model:      The unlearned model being evaluated (eval mode).
        ref_model:  Retain-only retrained reference model (for FQ).
                    Can be loaded from locuslab/tofu_ft_llama2-7b_retain95 etc.
        tokenizer:  Shared tokenizer.
        device:     "cuda" or "cpu".
        max_length: Tokenization truncation length.
        gen_tokens: Max new tokens for answer generation.
        n_probe:    Max examples per split to evaluate (speed / accuracy tradeoff).
    """

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        device: str | torch.device = "cuda",
        max_length: int = 256,
        gen_tokens: int = 30,
        n_probe:    int = 50,
    ):
        self.model      = model
        self.ref_model  = ref_model
        self.tokenizer  = tokenizer
        self.device     = torch.device(device)
        self.max_length = max_length
        self.gen_tokens = gen_tokens
        self.n_probe    = n_probe
        self._scorer    = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # ── Low-level helpers ─────────────────────────────────────────────────────
    @torch.no_grad()
    def _nll(self, model, text: str) -> float:
        """Per-token NLL of `text` under `model`."""
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        out = model(**enc, labels=enc["input_ids"])
        return float(out.loss.item())

    @torch.no_grad()
    def _generate(self, question: str) -> str:
        """Greedy generation given a question string."""
        prompt = f"Question: {question}\nAnswer:"
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=self.max_length).to(self.device)
        out = self.model.generate(
            **enc, max_new_tokens=self.gen_tokens,
            do_sample=False, pad_token_id=self.tokenizer.eos_token_id
        )
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return full[len(prompt):].strip()

    @torch.no_grad()
    def _token_log_prob(self, model, question: str, answer: str) -> float:
        """
        Average log-probability of `answer` tokens given `question`.
        Used for MIA-PR (likelihood ratio) and MIA-TR (truth ratio).
        """
        text = f"Question: {question}\nAnswer: {answer}"
        enc  = self.tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=self.max_length).to(self.device)
        out  = model(**enc)
        logits  = out.logits[..., :-1, :].contiguous()
        targets = enc["input_ids"][..., 1:].contiguous()
        log_p   = torch.nn.functional.log_softmax(logits, dim=-1)
        token_logp = log_p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(token_logp.mean().item())

    # ── Metric implementations ────────────────────────────────────────────────
    def forget_quality(
        self,
        forget_examples: Sequence[QAExample],
    ) -> float:
        """
        FQ: KS test p-value between NLL distributions of self.model and
        self.ref_model on the forget set.

        FQ → 1 means indistinguishable from the retain-only reference.
        FQ → 0 means the model still retains forget-set knowledge.
        """
        forget_subset = forget_examples[: self.n_probe]
        nll_student = []
        nll_ref     = []
        for ex in forget_subset:
            text = f"Question: {ex.question}\nAnswer: {ex.answer}"
            nll_student.append(self._nll(self.model,     text))
            nll_ref.append(    self._nll(self.ref_model, text))

        if len(nll_student) < 5:
            return 0.0
        _, p_value = stats.ks_2samp(nll_student, nll_ref)
        return float(p_value)

    def model_utility(
        self,
        retain_examples:   Sequence[QAExample],
        retain_ppl_ref:    float | None = None,
    ) -> float:
        """
        MU: composite of ROUGE-L on retain QA + normalised perplexity recovery.

        If retain_ppl_ref is None, uses self.ref_model as the baseline.
        """
        sub = retain_examples[: self.n_probe]

        # ROUGE-L component
        rouge_scores = []
        for ex in sub:
            gen  = self._generate(ex.question)
            s    = self._scorer.score(ex.answer, gen)["rougeL"].fmeasure
            rouge_scores.append(s)
        rouge_mean = float(np.mean(rouge_scores)) if rouge_scores else 0.0

        # Perplexity component
        ppls = [math.exp(self._nll(self.model, f"Question: {ex.question}\nAnswer: {ex.answer}"))
                for ex in sub]
        ppl_student = float(np.mean(ppls))

        if retain_ppl_ref is None:
            ppls_ref = [math.exp(self._nll(self.ref_model, f"Question: {ex.question}\nAnswer: {ex.answer}"))
                        for ex in sub]
            ppl_ref = float(np.mean(ppls_ref))
        else:
            ppl_ref = retain_ppl_ref

        ppl_score = math.exp(-(ppl_student - ppl_ref) / max(ppl_ref, 1e-6))
        ppl_score = float(np.clip(ppl_score, 0.0, 1.0))

        # Harmonic mean of ROUGE and ppl_score
        if rouge_mean + ppl_score < 1e-6:
            return 0.0
        return 2.0 * rouge_mean * ppl_score / (rouge_mean + ppl_score + 1e-6)

    def rouge_l(self, examples: Sequence[QAExample]) -> float:
        """Average ROUGE-L F1 over generated vs reference answers."""
        sub = examples[: self.n_probe]
        scores = []
        for ex in sub:
            gen = self._generate(ex.question)
            scores.append(self._scorer.score(ex.answer, gen)["rougeL"].fmeasure)
        return float(np.mean(scores)) if scores else 0.0

    def mia_truth_ratio(
        self,
        forget_examples: Sequence[QAExample],
        perturb_fn=None,
    ) -> float:
        """
        MIA Truth Ratio: ratio of log p(true answer) to log p(perturbed answer).

        truth_ratio = exp(log p(y_true) - log p(y_perturb))

        If perturb_fn is None, the reference answer is shuffled at word level.
        Values near 1.0 → model treats true/perturbed equally → good forgetting.
        High values (>> 1.0) → model still prefers the true answer → poor forgetting.
        """
        def default_perturb(answer: str) -> str:
            words = answer.split()
            if len(words) < 2:
                return answer + " [UNKWN]"
            np.random.shuffle(words)
            return " ".join(words)

        perturb = perturb_fn or default_perturb
        sub = forget_examples[: self.n_probe]
        ratios = []
        for ex in sub:
            lp_true    = self._token_log_prob(self.model, ex.question, ex.answer)
            lp_perturb = self._token_log_prob(self.model, ex.question, perturb(ex.answer))
            ratios.append(math.exp(lp_true - lp_perturb))
        return float(np.mean(ratios)) if ratios else 1.0

    def mia_probability_ratio(
        self,
        forget_examples: Sequence[QAExample],
    ) -> float:
        """
        MIA Probability Ratio: ratio of self.model to self.ref_model log probs.

        pr = exp(log p_θ(y|x) - log p_ref(y|x))

        pr → 1.0 → unlearned model assigns same probability as reference → good.
        pr >> 1.0 → unlearned model still prefers forget content → poor.
        """
        sub = forget_examples[: self.n_probe]
        ratios = []
        for ex in sub:
            text  = f"Question: {ex.question}\nAnswer: {ex.answer}"
            lp_s  = -self._nll(self.model,     text)
            lp_r  = -self._nll(self.ref_model, text)
            ratios.append(math.exp(lp_s - lp_r))
        return float(np.mean(ratios)) if ratios else 1.0

    # ── Full evaluation ───────────────────────────────────────────────────────
    def evaluate(
        self,
        forget_examples: Sequence[QAExample],
        retain_examples: Sequence[QAExample],
    ) -> MetricResult:
        """
        Run the complete 5-metric stack.

        Returns a MetricResult dataclass. Prints a one-line summary.
        """
        self.model.eval()
        self.ref_model.eval()

        print("  computing FQ...", end=" ", flush=True)
        fq = self.forget_quality(forget_examples)
        print(f"FQ={fq:.3f}")

        print("  computing MU...", end=" ", flush=True)
        mu = self.model_utility(retain_examples)
        print(f"MU={mu:.3f}")

        print("  computing ROUGE-L...", end=" ", flush=True)
        rl_f = self.rouge_l(forget_examples)
        rl_r = self.rouge_l(retain_examples)
        print(f"RL-f={rl_f:.3f}  RL-r={rl_r:.3f}")

        print("  computing MIA...", end=" ", flush=True)
        mia_tr = self.mia_truth_ratio(forget_examples)
        mia_pr = self.mia_probability_ratio(forget_examples)
        print(f"MIA-TR={mia_tr:.3f}  MIA-PR={mia_pr:.3f}")

        result = MetricResult(
            fq=fq, mu=mu,
            rouge_forget=rl_f, rouge_retain=rl_r,
            mia_tr=mia_tr, mia_pr=mia_pr,
        )
        print(f"  ✓  {result}")
        return result
