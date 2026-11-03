"""
forge/tofu_official.py — official-style TOFU evaluation metrics.

CROSS-VALIDATION NOTE (2026-07-30): the formulas below were corrected
against the original TOFU authors' reference implementation
(github.com/locuslab/tofu, `evaluate_util.py::eval_perturbation_ratio` and
`aggregate_eval_stat.py::get_forget_quality`/`get_model_utility`, cloned
and read directly for this pass). Three concrete bugs were found and fixed
relative to the first draft of this module:

  1. Truth Ratio aggregation over the k perturbed answers must be done in
     LOSS space (arithmetic mean of per-token NLL across perturbations,
     then a single exp of the difference), not by averaging probabilities
     computed separately per perturbation. These are not equivalent
     (Jensen's inequality: exp(mean(loss)) != mean(exp(loss))).
  2. The ratio direction was inverted: the reference computes
     exp(perturb_loss_mean - gt_loss), not exp(gt_loss - perturb_loss)
     divided differently -- the first draft computed the reciprocal.
  3. The Model Utility truth-ratio sub-score uses max(0, 1 - 1/R), not
     1 - min(R, 1) as the first draft guessed. The probability sub-score
     is RAW (mean of exp(-loss)) for the retain/forget sources but
     NORMALIZED (p_correct / (p_correct + sum(p_perturbed))) for the
     real_authors/world_facts sources -- this asymmetry is present in the
     reference code and is preserved here. ROUGE-L uses recall, not F1.

This is still an independent reimplementation (not the reference code
itself, and not run through the reference's exact generation/prompting
pipeline in all details) -- treat it as a validated-formula reproduction,
not a guaranteed byte-identical match, but it is now checked against
source, not guessed.

Definitions (per Maini et al., COLM 2024, arXiv:2401.06121, and the
reference implementation):

For an example with question q, correct answer a_c (paraphrased where
available, else canonical), and k perturbed/wrong answers {a_p^(k)}, let
ell(a|q) be the per-token mean NLL (teacher-forced) of a given q. Then:

    Truth Ratio R(x) = exp( mean_k[ ell(a_p^(k) | q) ]  -  ell(a_c | q_c) )

Official Forget Quality:
    FQ_official = KS-test p-value comparing {R(x) : x in D_forget} between
    the evaluated model and a retain-only oracle model.

Official Model Utility:
    MU_official = harmonic mean of 9 quantities: {ROUGE-L recall,
    answer probability, max(0, 1 - 1/R)} each computed over 3 data
    sources: {retain, real_authors, world_facts}. Probability is raw
    mean(exp(-ell)) for retain; normalized p_c/(p_c+sum(p_perturbed)) for
    real_authors/world_facts (matches the reference's branch on data source).

This module hard-fails (raises, does not silently substitute or invent
data) if any required TOFU field is missing from the loaded dataset.
"""
import math

import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer as rouge_lib
from scipy import stats as sc

REQUIRED_FORGET_FIELDS = {"question", "answer", "paraphrased_answer", "perturbed_answer", "paraphrased_question"}
REQUIRED_AUX_FIELDS = {"question", "answer", "perturbed_answer"}  # real_authors / world_facts (no paraphrase)


class MissingTOFUFieldError(RuntimeError):
    pass


def _check_fields(example, required, source_name):
    missing = required - set(example.keys())
    if missing:
        raise MissingTOFUFieldError(
            f"TOFU source '{source_name}' is missing required official-metric fields "
            f"{missing}. Refusing to invent these values — either use the correct "
            f"HF config (e.g. 'forget05_perturbed', not 'forget05') or fall back to "
            f"the FQ_NLL/MU_ret diagnostic metrics for this data source instead."
        )


@torch.no_grad()
def _seq_loss_normalized(model, tok, question, answer, device, max_length=256):
    """Length-normalized (per-token mean) NLL of `answer` given `question`,
    teacher-forced. Matches the reference's avg_gt_loss / avg_perturb_loss."""
    prompt = f"Question: {question}\nAnswer:"
    full = f"{prompt} {answer}"
    enc_prompt = tok(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    enc_full = tok(full, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    n_prompt_tokens = enc_prompt["input_ids"].shape[1]

    out = model(**enc_full)
    logits = out.logits[0, :-1, :]
    targets = enc_full["input_ids"][0, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    answer_logp = logp[n_prompt_tokens - 1:]
    if answer_logp.numel() == 0:
        return 100.0  # degenerate (empty answer span after truncation) -> huge loss
    return float(-answer_logp.mean().item())


def compute_answer_probability(model, tok, question, answer, device, max_length=256):
    """Raw normalized answer probability exp(-mean per-token NLL)."""
    loss = _seq_loss_normalized(model, tok, question, answer, device, max_length)
    return math.exp(-loss)


def compute_normalized_answer_probability(model, tok, example, device, max_length=256):
    """p_correct / (p_correct + sum(p_perturbed)) -- used for real_authors/world_facts
    in the reference's Model Utility 'Prob.' sub-score."""
    loss_c = _seq_loss_normalized(model, tok, example["question"], example["answer"], device, max_length)
    p_c = math.exp(-loss_c)
    p_perturbed = []
    for a_p in example["perturbed_answer"]:
        loss_p = _seq_loss_normalized(model, tok, example["question"], a_p, device, max_length)
        p_perturbed.append(math.exp(-loss_p))
    denom = p_c + sum(p_perturbed)
    return p_c / denom if denom > 0 else 0.0


def compute_truth_ratio(model, tok, example, device, max_length=256, has_paraphrase=True):
    """Truth Ratio R(x) = exp(mean_k[loss(a_p^(k)|q)] - loss(a_c|q_c)).
    Aggregation over k perturbed answers is in loss space (matches reference)."""
    q_c = example["paraphrased_question"] if has_paraphrase else example["question"]
    a_c = example["paraphrased_answer"] if has_paraphrase else example["answer"]
    loss_correct = _seq_loss_normalized(model, tok, q_c, a_c, device, max_length)

    perturbed_losses = []
    for a_p in example["perturbed_answer"]:
        loss_p = _seq_loss_normalized(model, tok, example["question"], a_p, device, max_length)
        perturbed_losses.append(loss_p)
    mean_perturbed_loss = sum(perturbed_losses) / len(perturbed_losses)

    return math.exp(mean_perturbed_loss - loss_correct)


def compute_rouge_l(model, tok, question, answer, device, max_new_tokens=30, max_length=200):
    """ROUGE-L RECALL (not F1) -- matches reference's rougeL_recall field."""
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    prompt = f"Question: {question}\nAnswer:"
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return scorer.score(answer, gen)["rougeL"].recall


def compute_forget_quality_tofu(model, oracle_model, tok, forget_perturbed_examples, device, max_length=256):
    """Official-style Forget Quality: KS-test on Truth Ratio distributions,
    model vs. retain-only oracle. Hard-fails if fields are missing."""
    for ex in forget_perturbed_examples:
        _check_fields(ex, REQUIRED_FORGET_FIELDS, "forget_perturbed")

    r_model = [compute_truth_ratio(model, tok, ex, device, max_length) for ex in forget_perturbed_examples]
    r_oracle = [compute_truth_ratio(oracle_model, tok, ex, device, max_length) for ex in forget_perturbed_examples]
    _, p = sc.ks_2samp(r_model, r_oracle)
    return float(p), r_model, r_oracle


def compute_model_utility_tofu(model, tok, retain_perturbed, real_authors_perturbed,
                                world_facts_perturbed, device, max_length=256, n_probe=30):
    """Official-style Model Utility: harmonic mean of 9 quantities across
    {retain, real_authors, world_facts} x {ROUGE-L recall, answer_prob, max(0,1-1/R)}.
    Hard-fails if required fields are missing from any of the three sources."""
    sources = [
        ("retain", retain_perturbed[:n_probe], True),
        ("real_authors", real_authors_perturbed[:n_probe], False),
        ("world_facts", world_facts_perturbed[:n_probe], False),
    ]
    values = []
    per_source = {}
    for name, examples, has_paraphrase in sources:
        req = REQUIRED_FORGET_FIELDS if has_paraphrase else REQUIRED_AUX_FIELDS
        for ex in examples:
            _check_fields(ex, req, name)

        rouges = [compute_rouge_l(model, tok, ex["question"], ex["answer"], device) for ex in examples]
        if has_paraphrase:
            # retain: raw probability, matches reference's 'eval_log' branch
            probs = [compute_answer_probability(model, tok, ex["question"], ex["answer"], device) for ex in examples]
        else:
            # real_authors / world_facts: normalized over correct + perturbed, matches reference
            probs = [compute_normalized_answer_probability(model, tok, ex, device, max_length) for ex in examples]
        truth_ratios = [compute_truth_ratio(model, tok, ex, device, max_length, has_paraphrase) for ex in examples]
        truth_scores = [max(0.0, 1.0 - 1.0 / r) if r > 0 else 0.0 for r in truth_ratios]

        rouge_mean = sum(rouges) / len(rouges)
        prob_mean = sum(probs) / len(probs)
        truth_mean = sum(truth_scores) / len(truth_scores)
        per_source[name] = {"rouge_l_recall": rouge_mean, "answer_prob": prob_mean, "truth_score": truth_mean}
        values += [rouge_mean, prob_mean, truth_mean]

    values = [max(v, 1e-8) for v in values]
    harmonic_mean = len(values) / sum(1.0 / v for v in values)
    return float(harmonic_mean), per_source
