# FORGE Method Summary

## Objective
ℒ(θ,g,ψ) = ℒ_forget + λ·ℒ_retain + γ·ℒ_robust(ψ) + ζ·ℒ_reason + η·ℒ_gate
           min over (θ,g);   max over ψ

## Components
- Gap A: Dual LoRA adapters (Δ_f, Δ_r) + Fisher-weighted merge + token gate g_φ(h)∈[0,1]
- Gap B: Adversarial inner loop; probe generator Φ_ψ updated k=2 steps before each outer step
- Gap C: Step-wise KL divergence on CoT traces (ζ=0 for TOFU; ζ=0.2 for R-TOFU)

## Default hyperparameters (RTX 3050 8 GB · Phi-3-mini-4B)
lr=5e-6, schedule=adaptive_4phase, rank=16, ep=10, λ=1.0, γ=0.5, η=0.01, k=2, ζ=0.0
max_steps=200, batch=1, grad_accum=4, max_length=128, quantization=4-bit
