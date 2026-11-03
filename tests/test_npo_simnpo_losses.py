"""
tests/test_npo_simnpo_losses.py — sanity tests for forge/trainer.py's
loss_NPO and loss_SimNPO (synced 2026-07-29 from the validated
scripts/run_experiments.py copies, see docs/experiment_audit.md).

Uses GPT-2 (smallest available cached model) + a tiny LoRA adapter purely
to keep these tests fast; they check loss-function contracts, not
TOFU-specific behavior (that's covered by the real GPU experiment outputs
in outputs/wk8/{NPO,SimNPO}/).

Run: python -m pytest tests/test_npo_simnpo_losses.py -q
Requires: GPU (skipped otherwise, consistent with this repo's GPU-first
policy — these losses are never exercised on CPU in real experiments).
"""
import pytest
import torch

peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from forge.trainer import loss_NPO, loss_SimNPO

MODEL_NAME = "gpt2"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required (repo policy: GPU-first, no CPU fallback for real losses)")


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda")


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    return tok


def _make_lora_model(device):
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
                     lora_dropout=0.0, target_modules=["c_attn"])
    model = get_peft_model(base, cfg).to(device)
    return model


def _make_frozen_ref(device):
    ref = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def _make_batch(tok, texts, device, batch_size=None):
    n = batch_size or len(texts)
    texts = texts[:n]
    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return {"input_ids": enc["input_ids"].to(device), "attention_mask": enc["attention_mask"].to(device), "labels": labels.to(device)}


FORGET_TEXTS = ["The quick brown fox jumps over the lazy dog.",
                "Paris is the capital of France.",
                "Water boils at one hundred degrees Celsius.",
                "The author was born in a small village."]
RETAIN_TEXTS = ["The sky is blue on a clear day.",
                "Mathematics is the study of numbers.",
                "Music can evoke strong emotions.",
                "Rivers flow toward the sea."]


class TestNPOLoss:
    def test_loss_is_finite(self, tokenizer, device):
        model = _make_lora_model(device)
        ref = _make_frozen_ref(device)
        forget = _make_batch(tokenizer, FORGET_TEXTS, device)
        retain = _make_batch(tokenizer, RETAIN_TEXTS, device)
        loss = loss_NPO(model, ref, forget, retain, device)
        assert torch.isfinite(loss), f"NPO loss is not finite: {loss}"

    def test_gradients_only_on_lora_params(self, tokenizer, device):
        model = _make_lora_model(device)
        ref = _make_frozen_ref(device)
        forget = _make_batch(tokenizer, FORGET_TEXTS, device)
        retain = _make_batch(tokenizer, RETAIN_TEXTS, device)
        loss = loss_NPO(model, ref, forget, retain, device)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert "lora" in name.lower(), f"Unexpected trainable non-LoRA param: {name}"
            else:
                assert p.grad is None, f"Frozen param unexpectedly has a gradient: {name}"

    def test_base_model_frozen(self, device):
        model = _make_lora_model(device)
        base_params = [p for n, p in model.named_parameters() if "lora" not in n.lower()]
        assert all(not p.requires_grad for p in base_params), "Base (non-LoRA) parameters must be frozen"

    def test_reference_model_frozen(self, device):
        ref = _make_frozen_ref(device)
        assert all(not p.requires_grad for p in ref.parameters()), "NPO reference model must be frozen"

    def test_runs_with_batch_size_1(self, tokenizer, device):
        model = _make_lora_model(device)
        ref = _make_frozen_ref(device)
        forget = _make_batch(tokenizer, FORGET_TEXTS, device, batch_size=1)
        retain = _make_batch(tokenizer, RETAIN_TEXTS, device, batch_size=1)
        loss = loss_NPO(model, ref, forget, retain, device)
        assert torch.isfinite(loss)
        loss.backward()  # should not raise


class TestSimNPOLoss:
    def test_loss_is_finite(self, tokenizer, device):
        model = _make_lora_model(device)
        forget = _make_batch(tokenizer, FORGET_TEXTS, device)
        retain = _make_batch(tokenizer, RETAIN_TEXTS, device)
        # SimNPO signature is (model, ref, forget, retain, device, ...) for API
        # symmetry with the other losses, but `ref` is unused (reference-free).
        loss = loss_SimNPO(model, None, forget, retain, device)
        assert torch.isfinite(loss), f"SimNPO loss is not finite: {loss}"

    def test_no_reference_model_required(self, tokenizer, device):
        """SimNPO must not touch its `ref` argument at all — passing a
        deliberately-wrong object should not raise or affect the loss."""
        model = _make_lora_model(device)
        forget = _make_batch(tokenizer, FORGET_TEXTS, device)
        retain = _make_batch(tokenizer, RETAIN_TEXTS, device)
        loss_a = loss_SimNPO(model, None, forget, retain, device)
        loss_b = loss_SimNPO(model, object(), forget, retain, device)  # nonsense ref, must be ignored
        assert torch.allclose(loss_a, loss_b), "SimNPO loss must be independent of the `ref` argument"

    def test_gradients_only_on_lora_params(self, tokenizer, device):
        model = _make_lora_model(device)
        forget = _make_batch(tokenizer, FORGET_TEXTS, device)
        retain = _make_batch(tokenizer, RETAIN_TEXTS, device)
        loss = loss_SimNPO(model, None, forget, retain, device)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert "lora" in name.lower(), f"Unexpected trainable non-LoRA param: {name}"

    def test_runs_with_batch_size_1(self, tokenizer, device):
        model = _make_lora_model(device)
        forget = _make_batch(tokenizer, FORGET_TEXTS, device, batch_size=1)
        retain = _make_batch(tokenizer, RETAIN_TEXTS, device, batch_size=1)
        loss = loss_SimNPO(model, None, forget, retain, device)
        assert torch.isfinite(loss)
        loss.backward()
