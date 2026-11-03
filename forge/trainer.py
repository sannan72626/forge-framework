"""
forge/trainer.py

STATUS NOTE (2026-07-29, docs/experiment_audit.md item 7/8): this module is
NOT currently imported by any script that produces real results — every
real GPU experiment in this repo (wk4/wk6/wk8/wk9, and all revision-phase
work) goes through the self-contained loss functions in
scripts/run_experiments.py instead. The original "Gap A/B/C" design below
(Fisher-weighted dual adapters, adversarial probe generator, CoT KL loss)
was an earlier, more ambitious proposal that was scoped down during
implementation; the actual validated FORGE objective is the simpler
bounded-ascent + retain + perturbed-forget-robustness loss documented on
`loss_FORGE` below, matching the IEEE IoT manuscript. The loss functions in
this file (including NPO/SimNPO) were synced to match
scripts/run_experiments.py in this revision pass for repo consistency, but
this file is not yet wired into the actual experiment pipeline — treat
scripts/run_experiments.py as canonical until this is resolved.

Original module description (superseded, kept for history):
FORGE UnlearnTrainer — drop-in replacement for OpenUnlearning's
base UnlearnTrainer.

Three novel components (NOT implemented / NOT part of the validated
objective — see loss_FORGE below for what is actually used):
  Gap A  Gated LoRA with Fisher-weighted adapter merge
  Gap B  Adversarial inner loop (probe generator Φ_ψ)
  Gap C  Step-wise CoT KL loss on reasoning traces (R-TOFU)

Dependencies:
  pip install transformers peft torch

Standalone:
  See forge_train_standalone() at the bottom.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Token-conditioned gate module
# ─────────────────────────────────────────────────────────────────────────────
class TokenGate(nn.Module):
    """
    Learns a per-token scalar gate g(h_t) ∈ [0, 1].

    At inference: forwards merged LoRA delta scaled by g.
    At training:  gate is regularised by L1 on retain-set hidden states,
                  pushing g → 0 on retain content.

    Architecture: 2-layer MLP  [hidden_dim → gate_hidden → 1 → sigmoid]
    """

    def __init__(self, hidden_dim: int, gate_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden, bias=True),
            nn.GELU(),
            nn.Linear(gate_hidden, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        """
        Args:
            hidden: [batch, seq_len, hidden_dim]
        Returns:
            gate:   [batch, seq_len, 1]  values in [0, 1]
        """
        return self.net(hidden)

    def sparsity_loss(self, retain_hidden: Tensor) -> Tensor:
        """L1 penalty on gate activations over retain inputs."""
        return self.forward(retain_hidden).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Fisher matrix (diagonal approximation)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_fisher(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int = 512,
) -> dict[str, Tensor]:
    """
    Compute diagonal Fisher information: F_i ≈ E[(∂L/∂θ_i)²].

    Returns a dict {param_name: fisher_diag_tensor}.
    Only parameters requiring grad are included.
    Called once before training; result is cached.
    """
    model.train()
    fisher: dict[str, Tensor] = {}
    count = 0

    for batch in dataloader:
        if count >= n_samples:
            break
        inputs = {k: v.to(device) for k, v in batch.items()
                  if isinstance(v, Tensor)}
        outputs = model(**inputs)
        loss = outputs.loss
        model.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_sq = param.grad.detach().pow(2)
                if name in fisher:
                    fisher[name] += grad_sq
                else:
                    fisher[name] = grad_sq.clone()

        count += inputs["input_ids"].shape[0]

    # Average
    for name in fisher:
        fisher[name] /= max(count, 1)

    return fisher


# ─────────────────────────────────────────────────────────────────────────────
# Adversary Φ_ψ — lightweight prefix-tuned probe generator
# ─────────────────────────────────────────────────────────────────────────────
class AdversaryProbeGenerator(nn.Module):
    """
    Lightweight adversary that generates hard paraphrase / perturb probes.

    Implementation: a learnable soft-prompt prefix injected before each
    forget-set query.  Only the prefix tokens (n_prefix × hidden_dim) are
    updated during the adversary step.  The base model is frozen.

    For a production-grade adversary, replace the prefix with a small
    seq2seq paraphraser (T5-small) — the interface stays identical.

    Args:
        model:      The FORGE student model (frozen during adversary steps).
        tokenizer:  Tokenizer (to decode / re-encode probes).
        n_prefix:   Number of learnable prefix tokens (default 20).
        hidden_dim: Embedding dimension of the base model.
    """

    def __init__(
        self,
        model: nn.Module,
        n_prefix: int = 20,
        hidden_dim: int = 4096,
    ):
        super().__init__()
        self.n_prefix   = n_prefix
        self.hidden_dim = hidden_dim
        # Learnable prefix embeddings — the only parameters updated for ψ
        self.prefix_embeds = nn.Parameter(
            torch.randn(1, n_prefix, hidden_dim) * 0.02
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        """
        Inject prefix into the forget-set batch; return augmented inputs
        as a dict ready for the student model forward pass.

        Returns:
            Dict with keys: inputs_embeds, attention_mask, labels
            (input_ids replaced by inputs_embeds to accommodate prefix)
        """
        bsz = input_ids.shape[0]
        # Expand prefix to batch
        prefix = self.prefix_embeds.expand(bsz, -1, -1)  # [B, n_prefix, H]
        return {
            "prefix_embeds":     prefix,
            "input_ids":         input_ids,
            "attention_mask":    attention_mask,
        }

    def adversary_step(
        self,
        student_model: nn.Module,
        forget_batch:  dict[str, Tensor],
        optimizer_adv: torch.optim.Optimizer,
        k: int = 2,
    ) -> Tensor:
        """
        Run k gradient-ascent steps on the adversary to maximise ℒ_robust.

        The student model parameters are frozen; only prefix_embeds update.

        Returns:
            ℒ_robust evaluated at the end of k steps (scalar).
        """
        student_model.eval()
        self.train()

        loss_robust = torch.zeros(1, device=self.prefix_embeds.device)
        for _ in range(k):
            # Get adversarially-augmented inputs
            aug = self.forward(
                forget_batch["input_ids"],
                forget_batch["attention_mask"],
            )
            # Forward through student (no grad through student weights)
            with torch.no_grad():
                out = student_model(
                    input_ids=aug["input_ids"],
                    attention_mask=aug["attention_mask"],
                    labels=forget_batch.get("labels", aug["input_ids"]),
                )
            # Adversary wants to MAXIMISE forget loss (reactivate content)
            loss_robust = -out.loss   # negated: maximise = minimise negative
            optimizer_adv.zero_grad()
            loss_robust.backward()
            optimizer_adv.step()

        student_model.train()
        return loss_robust.detach()


# ─────────────────────────────────────────────────────────────────────────────
# FORGE training arguments
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FORGEArgs:
    # LoRA
    lora_rank:        int   = 32
    lora_alpha:       int   = 64
    lora_dropout:     float = 0.0
    lora_target:      str   = "q_proj,v_proj"   # comma-separated

    # Loss weights
    lambda_retain: float = 1.0
    gamma_robust:  float = 0.5
    zeta_reason:   float = 0.0   # > 0 for R-TOFU CoT loss
    eta_gate:      float = 0.01  # gate sparsity on retain

    # Adversary
    k_adv:         int   = 2
    n_prefix:      int   = 20
    lr_adv:        float = 1e-4  # adversary prefix lr (separate from model lr)

    # Fisher
    fisher_samples: int  = 512

    # Gate
    gate_hidden:   int   = 64

    # Misc
    ihl_margin: float = 4.0   # Inverted Hinge Loss clip margin


# ─────────────────────────────────────────────────────────────────────────────
# FORGE Trainer
# ─────────────────────────────────────────────────────────────────────────────
class FORGETrainer(Trainer):
    """
    FORGE unlearning trainer.

    Drop-in replacement for HuggingFace Trainer (or OpenUnlearning's
    UnlearnTrainer).  Overrides compute_loss() to inject the 5-term objective.

    Setup (called once before training.train()):
        trainer.setup_forge(forget_dataloader, retain_dataloader)

    After setup, call trainer.train() normally.
    """

    def __init__(self, *args, forge_args: FORGEArgs | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.forge_args = forge_args or FORGEArgs()
        self._forge_ready = False

    # ── One-time setup ────────────────────────────────────────────────────────
    def setup_forge(
        self,
        forget_dataloader: torch.utils.data.DataLoader,
        retain_dataloader: torch.utils.data.DataLoader,
        ref_model:         Optional[nn.Module] = None,
    ) -> None:
        """
        Must be called once before trainer.train().

        Args:
            forget_dataloader: DataLoader over D_f (forget set).
            retain_dataloader: DataLoader over D_r (retain set).
            ref_model:         Frozen pre-unlearn model for ℒ_reason.
                               If None, a deepcopy of self.model is used.
        """
        fa = self.forge_args
        device = next(self.model.parameters()).device

        # (1) Wrap model with two separate LoRA adapters
        lora_targets = [t.strip() for t in fa.lora_target.split(",")]
        lora_cfg_f = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=fa.lora_rank, lora_alpha=fa.lora_alpha,
            lora_dropout=fa.lora_dropout,
            target_modules=lora_targets,
        )
        self.model = get_peft_model(self.model, lora_cfg_f)
        logger.info("Added forget LoRA adapter (rank=%d)", fa.lora_rank)

        # (2) Compute Fisher information (one pass, cached)
        logger.info("Computing Fisher matrices over %d samples...", fa.fisher_samples)
        self.fisher_f = compute_fisher(
            self.model, forget_dataloader, device, n_samples=fa.fisher_samples
        )
        self.fisher_r = compute_fisher(
            self.model, retain_dataloader, device, n_samples=fa.fisher_samples
        )
        logger.info("Fisher matrices ready.")

        # (3) Token gate
        hidden_dim = self.model.config.hidden_size
        self.gate = TokenGate(hidden_dim, gate_hidden=fa.gate_hidden).to(device)
        logger.info("Gate module: hidden=%d → %d → 1", hidden_dim, fa.gate_hidden)

        # (4) Adversary
        self.adversary = AdversaryProbeGenerator(
            model=self.model,
            n_prefix=fa.n_prefix,
            hidden_dim=hidden_dim,
        ).to(device)
        self.optimizer_adv = torch.optim.Adam(
            self.adversary.parameters(), lr=fa.lr_adv
        )
        logger.info("Adversary prefix length=%d, lr=%.2e", fa.n_prefix, fa.lr_adv)

        # (5) Reference model for ℒ_reason
        if fa.zeta_reason > 0:
            self.ref_model = (
                ref_model if ref_model is not None else copy.deepcopy(self.model)
            )
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
            logger.info("Reference model ready for ℒ_reason.")
        else:
            self.ref_model = None

        # (6) Iterators for forget/retain batches
        self._forget_loader  = forget_dataloader
        self._retain_loader  = retain_dataloader
        self._forget_iter    = iter(forget_dataloader)
        self._retain_iter    = iter(retain_dataloader)

        self._forge_ready = True
        logger.info("FORGE setup complete.")

    # ── Batch sampling helpers ────────────────────────────────────────────────
    def _next_forget(self) -> dict[str, Tensor]:
        try:
            return next(self._forget_iter)
        except StopIteration:
            self._forget_iter = iter(self._forget_loader)
            return next(self._forget_iter)

    def _next_retain(self) -> dict[str, Tensor]:
        try:
            return next(self._retain_iter)
        except StopIteration:
            self._retain_iter = iter(self._retain_loader)
            return next(self._retain_iter)

    # ── Loss components ───────────────────────────────────────────────────────
    def _forget_loss(self, model: nn.Module, forget_batch: dict) -> Tensor:
        """
        ℒ_forget: Inverted Hinge Loss (IHL).
        Negates NLL on D_f but clips at ±ihl_margin to prevent
        unbounded divergence (plain gradient ascent is numerically unstable).
        """
        M = self.forge_args.ihl_margin
        out = model(**{k: v for k, v in forget_batch.items() if k != "is_forget"})
        nll = out.loss
        # IHL: -tanh(nll/M)*M  — bounded gradient ascent
        return -torch.tanh(nll / M) * M

    def _retain_loss(self, model: nn.Module, retain_batch: dict) -> Tensor:
        """ℒ_retain: standard NLL on D_r."""
        out = model(**{k: v for k, v in retain_batch.items() if k != "is_forget"})
        return out.loss

    def _robust_loss(
        self, model: nn.Module, forget_batch: dict, device: torch.device
    ) -> Tensor:
        """
        ℒ_robust: forget loss on adversarially-generated probes x̃.

        The adversary inner loop runs k steps first (updates ψ),
        then we evaluate the student on the resulting probes.
        """
        fa = self.forge_args

        # Step adversary (k inner steps) — updates ψ, not θ
        self.adversary.adversary_step(
            student_model=model,
            forget_batch={k: v.to(device) for k, v in forget_batch.items()},
            optimizer_adv=self.optimizer_adv,
            k=fa.k_adv,
        )

        # Evaluate student on adversarial probes (updates θ, not ψ)
        aug = self.adversary(
            input_ids=forget_batch["input_ids"].to(device),
            attention_mask=forget_batch["attention_mask"].to(device),
        )
        out = model(
            input_ids=aug["input_ids"],
            attention_mask=aug["attention_mask"],
            labels=forget_batch.get("labels", aug["input_ids"]).to(device),
        )
        return out.loss   # student minimises this

    def _gate_sparsity_loss(self, retain_batch: dict, device: torch.device) -> Tensor:
        """
        ℒ_gate: L1 sparsity penalty on gate activations over retain inputs.
        Pushes g_φ(h_r) → 0, enforcing binary (not soft) routing.
        """
        # We need the hidden states from the base model
        with torch.no_grad():
            out = self.model(
                input_ids=retain_batch["input_ids"].to(device),
                attention_mask=retain_batch["attention_mask"].to(device),
                output_hidden_states=True,
            )
        # Use the last hidden layer as gate input
        hidden = out.hidden_states[-1]  # [B, T, H]
        return self.gate.sparsity_loss(hidden)

    def _reason_loss(self, model: nn.Module, forget_batch: dict, device: torch.device) -> Tensor:
        """
        ℒ_reason: step-wise KL divergence on CoT reasoning traces.
        Σ_k KL(p_θ(z_k | z_{<k}, x) ‖ p_ref(z_k | z_{<k}, x))

        Assumes forget_batch contains "cot_input_ids" and "cot_attention_mask"
        keys for the chain-of-thought prefix (R-TOFU format).
        If these keys are absent, falls back to the canonical query.
        """
        if self.ref_model is None:
            return torch.zeros(1, device=device)

        ids  = forget_batch.get("cot_input_ids",
               forget_batch["input_ids"]).to(device)
        mask = forget_batch.get("cot_attention_mask",
               forget_batch["attention_mask"]).to(device)

        with torch.no_grad():
            ref_out = self.ref_model(input_ids=ids, attention_mask=mask)
        student_out = model(input_ids=ids, attention_mask=mask)

        # KL per token position, averaged over sequence and batch
        student_log_probs = F.log_softmax(student_out.logits, dim=-1)
        ref_probs         = F.softmax(ref_out.logits.detach(), dim=-1)
        kl = F.kl_div(student_log_probs, ref_probs, reduction="batchmean")
        return kl

    # ── Main compute_loss override ────────────────────────────────────────────
    def compute_loss(
        self,
        model:          nn.Module,
        inputs:         dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Override of HuggingFace Trainer.compute_loss.
        Called by Trainer.training_step() at every step.
        """
        if not self._forge_ready:
            raise RuntimeError("Call setup_forge() before training.")

        fa     = self.forge_args
        device = next(model.parameters()).device

        forget_batch = self._next_forget()
        retain_batch = self._next_retain()

        # ── Gap A: forget loss (IHL) ─────────────────────────────────────────
        l_forget = self._forget_loss(model, {k: v.to(device) for k, v in forget_batch.items()})

        # ── Gap A: retain loss ───────────────────────────────────────────────
        l_retain = self._retain_loss(model, {k: v.to(device) for k, v in retain_batch.items()})

        # ── Gap B: adversarial robust loss ───────────────────────────────────
        l_robust = (
            self._robust_loss(model, forget_batch, device)
            if fa.gamma_robust > 0 else torch.zeros(1, device=device)
        )

        # ── Gap A: gate sparsity ─────────────────────────────────────────────
        l_gate = (
            self._gate_sparsity_loss(retain_batch, device)
            if fa.eta_gate > 0 else torch.zeros(1, device=device)
        )

        # ── Gap C: CoT step-wise KL ──────────────────────────────────────────
        l_reason = (
            self._reason_loss(model, forget_batch, device)
            if fa.zeta_reason > 0 else torch.zeros(1, device=device)
        )

        # ── Total ────────────────────────────────────────────────────────────
        loss = (
            l_forget
            + fa.lambda_retain * l_retain
            + fa.gamma_robust  * l_robust
            + fa.eta_gate      * l_gate
            + fa.zeta_reason   * l_reason
        )

        if return_outputs:
            return loss, {
                "loss_forget": l_forget.item(),
                "loss_retain": l_retain.item(),
                "loss_robust": l_robust.item(),
                "loss_gate":   l_gate.item(),
                "loss_reason": l_reason.item(),
                "loss_total":  loss.item(),
            }
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Standalone training helper (no OpenUnlearning dependency)
# ─────────────────────────────────────────────────────────────────────────────
def forge_train_standalone(
    model_name:        str = "locuslab/tofu_ft_llama2-7b",
    forget_split:      str = "forget05",
    retain_split:      str = "retain95",
    output_dir:        str = "outputs/forge_run",
    total_steps:       int = 300,
    lr:                float = 5e-6,
    per_device_batch:  int = 4,
    forge_args:        FORGEArgs | None = None,
) -> None:
    """
    Convenience wrapper for standalone (non-OpenUnlearning) training.

    Requires:
        pip install transformers peft datasets
        # GPU with ≥24 GB (fp16) or ≥80 GB (bf16 + Llama-2-7B full)

    Example:
        python -c "from forge.trainer import forge_train_standalone; forge_train_standalone()"
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
    from datasets import load_dataset
    from forge.adaptive_lr import FORGEScheduler

    fa = forge_args or FORGEArgs()

    # Model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Data
    def tokenize(batch):
        texts = [q + " " + a for q, a in zip(batch["question"], batch["answer"])]
        enc   = tokenizer(texts, truncation=True, max_length=256, padding="max_length")
        enc["labels"] = enc["input_ids"].copy()
        return enc

    forget_ds = load_dataset("locuslab/TOFU", forget_split)["train"].map(tokenize, batched=True)
    retain_ds = load_dataset("locuslab/TOFU", retain_split)["train"].map(tokenize, batched=True)
    forget_ds.set_format("torch")
    retain_ds.set_format("torch")

    forget_dl = torch.utils.data.DataLoader(forget_ds, batch_size=per_device_batch, shuffle=True)
    retain_dl = torch.utils.data.DataLoader(retain_ds, batch_size=per_device_batch, shuffle=True)

    # Training args
    train_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=total_steps,
        per_device_train_batch_size=per_device_batch,
        learning_rate=lr,
        lr_scheduler_type="constant",   # FORGE uses its own scheduler below
        save_steps=50,
        logging_steps=10,
        bf16=True,
        report_to="none",
    )

    # Trainer
    trainer = FORGETrainer(
        model=model,
        args=train_args,
        train_dataset=forget_ds,
        forge_args=fa,
    )

    # Inject adaptive LR scheduler after optimizer creation
    trainer.create_optimizer()
    forge_sched = FORGEScheduler(trainer.optimizer, total_steps=total_steps, lr_peak=lr)
    trainer.lr_scheduler = forge_sched

    # Setup FORGE components
    trainer.setup_forge(forget_dl, retain_dl)

    # Train
    logger.info("Starting FORGE training for %d steps...", total_steps)
    trainer.train()
    trainer.save_model(output_dir)
    logger.info("Training complete. Model saved to %s", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone loss functions (used by run_experiments.py and stats.py)
# ─────────────────────────────────────────────────────────────────────────────
def _nll(model, batch, device):
    return model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    ).loss

def loss_GA(model, ref, forget, retain, device):
    return -_nll(model, forget, device)

def loss_GD(model, ref, forget, retain, device):
    l_f = -_nll(model, forget, device)
    l_rs = _nll(model, retain, device)
    with torch.no_grad():
        l_rr = _nll(ref, retain, device)
    return l_f + (l_rs - l_rr).abs()

def loss_KL(model, ref, forget, retain, device):
    l_f = -_nll(model, forget, device)
    out_s = model(input_ids=retain["input_ids"].to(device),
                  attention_mask=retain["attention_mask"].to(device))
    with torch.no_grad():
        out_r = ref(input_ids=retain["input_ids"].to(device),
                    attention_mask=retain["attention_mask"].to(device))
    kl = F.kl_div(F.log_softmax(out_s.logits, -1),
                  F.softmax(out_r.logits.detach(), -1), reduction="batchmean")
    return l_f + kl

def loss_NPO(model, ref, forget, retain, device, beta=2.5):
    # beta default synced 2026-07-29 with scripts/run_experiments.py::loss_NPO
    # (the module actually used for all real GPU results in this repo — this
    # copy in forge/trainer.py is not imported by any working script, see
    # docs/experiment_audit.md item 7). The original beta=0.1 here amplifies
    # the loss by 2/beta=20x under the shared lr=2e-5 and causes the same
    # catastrophic collapse documented for the run_experiments.py copy
    # (outputs/probe/npo_beta0.1 equivalent trajectories); beta=2.5 keeps the
    # peak joint FQ_NLL*MU_ret operating point intact for a checkpoint-selected
    # evaluation instead of overshooting past it.
    out_s = model(input_ids=forget["input_ids"].to(device),
                  attention_mask=forget["attention_mask"].to(device))
    with torch.no_grad():
        out_r = ref(input_ids=forget["input_ids"].to(device),
                    attention_mask=forget["attention_mask"].to(device))
    tgts = forget["labels"].to(device)[..., 1:]
    mask = (tgts != -100).float()
    tgt_c = tgts.masked_fill(tgts == -100, 0)
    logp = F.log_softmax(out_s.logits[..., :-1, :], -1).gather(-1, tgt_c.unsqueeze(-1)).squeeze(-1)
    rlogp = F.log_softmax(out_r.logits[..., :-1, :], -1).gather(-1, tgt_c.unsqueeze(-1)).squeeze(-1)
    log_ratio = ((logp - rlogp) * mask).sum(-1) / (mask.sum(-1) + 1e-8)
    return (-2 / beta * torch.log(torch.sigmoid(-beta * log_ratio) + 1e-8)).mean()


def loss_SimNPO(model, ref, forget, retain, device, beta=2.5, gamma=5.0, lam=1.0):
    """SimNPO (Fan et al., NeurIPS 2025, arXiv:2410.07163) — reference-free,
    length-normalized, margin-scaled saturating ascent. Synced 2026-07-29 from
    scripts/run_experiments.py::loss_SimNPO (the module actually used for real
    GPU results); this copy was previously entirely absent from forge/trainer.py.
    gamma=5.0 default found empirically: gamma=0 (the naive choice) produces
    zero forgetting effect since the model's own forget-set NLL starts small
    (already-memorized data has low NLL), so beta*u-gamma with gamma=0 is
    already past the saturation point from step 0. gamma=5.0 shifts the
    starting point into the unsaturated region (see outputs/probe/simnpo_gamma*
    for the sweep that produced this value).
    """
    out = model(input_ids=forget["input_ids"].to(device),
                attention_mask=forget["attention_mask"].to(device))
    tgts = forget["labels"].to(device)[..., 1:]
    mask = (tgts != -100).float()
    tgt_c = tgts.masked_fill(tgts == -100, 0)
    logp = F.log_softmax(out.logits[..., :-1, :], -1).gather(-1, tgt_c.unsqueeze(-1)).squeeze(-1)
    u = -((logp * mask).sum(-1) / (mask.sum(-1) + 1e-8))  # length-normalized forget NLL per sample
    l_f = (-2.0 / beta * torch.log(torch.sigmoid(beta * u - gamma) + 1e-8)).mean()
    l_r = _nll(model, retain, device)
    return l_f + lam * l_r

def loss_FLAT(model, ref, forget, retain, device, beta=2.0):
    out = model(input_ids=forget["input_ids"].to(device),
                attention_mask=forget["attention_mask"].to(device))
    tgts = forget["labels"].to(device)[..., 1:]
    mask = (tgts != -100).float()
    tgt_c = tgts.masked_fill(tgts == -100, 0)
    logp = F.log_softmax(out.logits[..., :-1, :], -1).gather(-1, tgt_c.unsqueeze(-1)).squeeze(-1)
    p = logp.exp()
    flat_ = -torch.log(1.0 + beta * p + 1e-8)
    return (flat_ * mask).sum() / (mask.sum() + 1e-8)

def loss_LoKU(model, ref, forget, retain, device, margin=0.5, lam=1.0):
    l_f = -torch.clamp(_nll(model, forget, device), max=margin)
    l_r = _nll(model, retain, device)
    return l_f + lam * l_r

def loss_FORGE(model, ref, forget, retain, device,
               lam=1.0, gamma=0.5, ihl_margin=4.0):
    """FORGE: bounded-ascent unlearning with input-perturbation robustness.

    Canonical objective (matches scripts/run_experiments.py, the version that
    produced all reported results):

        ℒ_FORGE = ℒ_forget + λ·ℒ_retain + γ·ℒ_robust
          ℒ_forget = −tanh(NLL_f / m) · m            # bounded gradient ascent
          ℒ_retain =  NLL(retain)                    # utility anchor
          ℒ_robust = −tanh(NLL(perturb(forget)) / m) · m  # ascent on perturbed forget

    The tanh bound caps the per-step forget gradient, preventing the runaway
    divergence that collapses utility in plain GA/GD. The robustness term forces
    forgetting to hold under input perturbation (random 20% token masking), a
    lightweight retain-free robustness signal.

    NOTE: the originally-proposed Fisher-weighted dual adapters, learned adversarial
    probe generator, and CoT-trace KL are scoped as future work (see
    outputs/reports/method_audit.md) and are NOT part of this objective.
    """
    # ℒ_forget — bounded gradient ascent (IHL-style)
    nll_f = _nll(model, forget, device)
    l_f = -torch.tanh(nll_f / ihl_margin) * ihl_margin

    # ℒ_retain — utility anchor
    l_r = _nll(model, retain, device)

    # ℒ_robust — bounded ascent on input-perturbed forget (NOT minimization)
    ids_rob = forget["input_ids"].clone().to(device)
    ids_rob[torch.rand_like(ids_rob.float()) < 0.20] = 0
    nll_rob = model(input_ids=ids_rob,
                    attention_mask=forget["attention_mask"].to(device),
                    labels=forget["labels"].to(device)).loss
    l_rob = -torch.tanh(nll_rob / ihl_margin) * ihl_margin

    return l_f + lam * l_r + gamma * l_rob


def loss_FORGEH(model, ref, forget, retain, device, margin=4.0, lam=1.0, gamma=0.5):
    """FORGE-Hybrid: LoKU hard-clamp forget + clamped perturbation-robustness.

    Keeps LoKU's hard clamp (stable at scale — no overshoot) and adds a clamped
    robustness term on input-perturbed forget. See outputs/reports/llama2_7b_report.md
    for the diagnosis motivating this design.
    """
    l_f = -torch.clamp(_nll(model, forget, device), max=margin)
    l_r = _nll(model, retain, device)
    ids_rob = forget["input_ids"].clone().to(device)
    ids_rob[torch.rand_like(ids_rob.float()) < 0.20] = 0
    nll_rob = model(input_ids=ids_rob,
                    attention_mask=forget["attention_mask"].to(device),
                    labels=forget["labels"].to(device)).loss
    l_rob = -torch.clamp(nll_rob, max=margin)
    return l_f + lam * l_r + gamma * l_rob


METHOD_LOSSES = {
    "GA":     loss_GA,
    "GD":     loss_GD,
    "KL":     loss_KL,
    "NPO":    loss_NPO,
    "SimNPO": loss_SimNPO,
    "FLAT":   loss_FLAT,
    "LoKU":   loss_LoKU,
    "FORGE":  loss_FORGE,
    "FORGEH": loss_FORGEH,
}
