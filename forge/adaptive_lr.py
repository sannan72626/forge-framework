"""
forge/adaptive_lr.py
────────────────────
4-phase adaptive learning-rate scheduler for FORGE.

Phases (all configurable):
  1. Linear warm-up     (steps 0 .. t_warm)
     Ramps lr from lr_base to lr_peak.
     Purpose: let the token gate g_φ learn routing signal
     before large forget-adapter updates happen.

  2. Cosine decay       (steps t_warm .. t_decay)
     Decays lr from lr_peak back to lr_base.
     Purpose: drive FQ with high lr early, then protect MU
     as the forget/retain boundary tightens.

  3. Adversarial plateau (steps t_decay .. t_plat)
     Holds lr at lr_adv (slightly above lr_base).
     Purpose: give the adversary Φ_ψ enough signal to
     generate hard probes without disturbing the main model.

  4. Cosine cool-down   (steps t_plat .. t_total)
     Decays lr from lr_adv to lr_floor.
     Purpose: stabilise gate scores and avoid small oscillations
     near the forget boundary at the end of training.

Usage:
    from forge.adaptive_lr import FORGEScheduler

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    scheduler = FORGEScheduler(optimizer, total_steps=300)

    for step in range(300):
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Current lr at any step:
    scheduler.get_last_lr()
"""

import math
import warnings
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class FORGEScheduler(LRScheduler):
    """
    4-phase lr schedule for FORGE unlearning.

    All step boundaries are given as fractions of total_steps if they are
    floats in (0, 1), or as absolute step counts if they are ints > 1.

    Args:
        optimizer:    PyTorch optimizer (typically AdamW).
        total_steps:  Total number of training steps.
        lr_peak:      Peak lr reached at end of warm-up phase.
                      Defaults to the initial lr in the optimizer.
        lr_base:      lr at start of training and at end of cosine decay.
                      Defaults to lr_peak * 0.5.
        lr_adv:       lr during the adversarial plateau.
                      Defaults to lr_peak * 0.7.
        lr_floor:     Minimum lr at end of cool-down.
                      Defaults to lr_peak * 0.2.
        t_warm:       End of warm-up phase (fraction or steps). Default 0.10.
        t_decay:      End of cosine decay phase (fraction or steps). Default 0.67.
        t_plat:       End of adversarial plateau (fraction or steps). Default 0.83.
        last_epoch:   Standard PyTorch LRScheduler arg.
        verbose:      Print lr changes.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        lr_peak: float | None = None,
        lr_base: float | None = None,
        lr_adv:  float | None = None,
        lr_floor: float | None = None,
        t_warm:  float | int = 0.10,
        t_decay: float | int = 0.67,
        t_plat:  float | int = 0.83,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        # Resolve lr_peak from optimizer if not given
        if lr_peak is None:
            lr_peak = optimizer.defaults["lr"]

        self.lr_peak  = lr_peak
        self.lr_base  = lr_base  if lr_base  is not None else lr_peak * 0.5
        self.lr_adv   = lr_adv   if lr_adv   is not None else lr_peak * 0.7
        self.lr_floor = lr_floor if lr_floor is not None else lr_peak * 0.2

        self.total_steps = total_steps

        # Resolve fractional boundaries to absolute step counts
        def _resolve(v):
            if isinstance(v, float) and 0.0 < v < 1.0:
                return max(1, int(round(v * total_steps)))
            elif isinstance(v, int) and v > 1:
                return v
            else:
                raise ValueError(f"Phase boundary must be a float in (0,1) or int > 1; got {v!r}")

        self.t_warm  = _resolve(t_warm)
        self.t_decay = _resolve(t_decay)
        self.t_plat  = _resolve(t_plat)

        if not (self.t_warm < self.t_decay < self.t_plat < total_steps):
            raise ValueError(
                f"Phase boundaries must satisfy t_warm < t_decay < t_plat < total_steps. "
                f"Got {self.t_warm}, {self.t_decay}, {self.t_plat}, {total_steps}."
            )

        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def _compute_lr(self, step: int) -> float:
        """Compute lr at an absolute step."""
        if step <= self.t_warm:
            # Phase 1: linear warm-up from lr_base → lr_peak
            return self.lr_base + (self.lr_peak - self.lr_base) * (step / self.t_warm)

        elif step <= self.t_decay:
            # Phase 2: cosine decay from lr_peak → lr_base
            progress = (step - self.t_warm) / (self.t_decay - self.t_warm)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.lr_base + (self.lr_peak - self.lr_base) * cosine

        elif step <= self.t_plat:
            # Phase 3: adversarial plateau at lr_adv
            return self.lr_adv

        else:
            # Phase 4: cosine cool-down from lr_adv → lr_floor
            progress = (step - self.t_plat) / (self.total_steps - self.t_plat)
            progress = min(progress, 1.0)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.lr_floor + (self.lr_adv - self.lr_floor) * cosine

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        lr   = self._compute_lr(step)
        return [lr for _ in self.base_lrs]

    # ── Convenience ──────────────────────────────────────────────────────────
    @property
    def phase(self) -> str:
        """Return name of current phase for logging."""
        step = self.last_epoch
        if step <= self.t_warm:
            return "warmup"
        elif step <= self.t_decay:
            return "cosine_decay"
        elif step <= self.t_plat:
            return "adv_plateau"
        else:
            return "cooldown"

    def get_schedule_values(self, n_steps: int | None = None) -> list[float]:
        """Return lr at every step (useful for visualisation / unit tests)."""
        n = n_steps or self.total_steps
        return [self._compute_lr(t) for t in range(n)]

    def __repr__(self) -> str:
        return (
            f"FORGEScheduler("
            f"total_steps={self.total_steps}, "
            f"lr_peak={self.lr_peak:.2e}, lr_base={self.lr_base:.2e}, "
            f"lr_adv={self.lr_adv:.2e}, lr_floor={self.lr_floor:.2e}, "
            f"phases=[0→{self.t_warm}|{self.t_warm}→{self.t_decay}|"
            f"{self.t_decay}→{self.t_plat}|{self.t_plat}→{self.total_steps}])"
        )


# ── Quick sanity plot ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch, matplotlib.pyplot as plt

    dummy_model = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(dummy_model.parameters(), lr=1e-5)
    sched = FORGEScheduler(opt, total_steps=300)
    print(sched)

    lrs = sched.get_schedule_values()
    plt.figure(figsize=(9, 4))
    plt.plot(lrs, color="#1E2761", lw=2.5)
    for boundary, label in [
        (sched.t_warm,  "← warm-up ends"),
        (sched.t_decay, "← decay ends"),
        (sched.t_plat,  "← plateau ends"),
    ]:
        plt.axvline(boundary, color="#BBB", ls="--", lw=1.0)
        plt.text(boundary+2, max(lrs)*0.98, label, fontsize=8)
    plt.yscale("log")
    plt.xlabel("Training step"); plt.ylabel("Learning rate (log scale)")
    plt.title("FORGEScheduler — 4-phase adaptive LR")
    plt.tight_layout(); plt.show()
