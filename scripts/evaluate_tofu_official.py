#!/usr/bin/env python
"""
scripts/evaluate_tofu_official.py — official-style TOFU FQ/MU for one
trained checkpoint. See forge/tofu_official.py for the honesty caveat:
this is an independent reimplementation, not verified byte-identical to
the original TOFU authors' reference code.

Evidence-remediation fix (Phase 1, task 3): this script now (a) records the
sha256 of the actual adapter weight file it evaluated, so the output JSON
is independently self-describing rather than trusting the `--adapter` path
string alone, and (b) optionally cross-checks that path/hash against a
train_method() checkpoint manifest, hard-failing if the official-metric
evaluation would not be looking at the same checkpoint the diagnostic
FQ_NLL/MU_ret numbers were selected from.

Usage (pass the `_selected` checkpoint, NOT the old unconditional one):
  python scripts/evaluate_tofu_official.py \
    --model locuslab/tofu_ft_phi-1.5 --base_model microsoft/phi-1_5 \
    --adapter outputs/.../FORGE_selected \
    --oracle_adapter outputs/wk0/phi15_retain95 \
    --split forget05 \
    --checkpoint_manifest outputs/.../checkpoint_manifests.json --method FORGE \
    --output_json outputs/official_tofu/FORGE/forget05/seed42.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from forge.tofu_official import (
    compute_forget_quality_tofu,
    compute_model_utility_tofu,
    MissingTOFUFieldError,
)

TOK_MODEL = "microsoft/phi-1_5"


def _hash_adapter_dir(adapter_path):
    for name in ("adapter_model.safetensors", "model.pt", "adapter_model.bin"):
        p = Path(adapter_path) / name
        if p.exists():
            h = hashlib.sha256()
            h.update(p.read_bytes())
            return h.hexdigest()
    return None


def verify_checkpoint_identity(adapter_sha256, checkpoint_manifest_path, method):
    """Pure identity check extracted from main() (Phase 1, task 6) so it can be
    unit-tested without GPU/model loading. Returns the manifest's recorded
    selected_step on success; raises RuntimeError on any mismatch."""
    manifests = json.loads(Path(checkpoint_manifest_path).read_text())
    m = manifests.get(method)
    if m is None:
        raise RuntimeError(f"Method '{method}' not found in {checkpoint_manifest_path}")
    expected_hash = m.get("selected_checkpoint_sha256")
    expected_selected_step = m.get("selected_step")
    if expected_hash != adapter_sha256:
        raise RuntimeError(
            f"CHECKPOINT IDENTITY MISMATCH: --adapter has sha256 "
            f"{adapter_sha256}, but the checkpoint manifest for method='{method}' "
            f"records selected_checkpoint_sha256={expected_hash} (selected_step="
            f"{expected_selected_step}). Refusing to evaluate official metrics on a "
            f"checkpoint that does not match the diagnostic-selected one. Pass the "
            f"'{method}_selected' directory, not a different checkpoint."
        )
    return expected_selected_step


def _adapter_base_model(adapter_path, fallback_base_model):
    """Phase 2A finding (Blocker D): different adapters in this project were
    trained from DIFFERENT base models -- e.g. the retain-oracle from plain
    microsoft/phi-1_5, but the method adapters (GA/.../FORGE) from
    locuslab/tofu_ft_phi-1.5. A single --base_model applied to both legs (the
    prior behavior here) silently merges the method adapter onto the WRONG
    base, producing a model unrelated to what was actually trained. Read each
    adapter's own recorded base from its adapter_config.json instead of
    trusting one CLI flag for everything; --base_model remains the fallback
    for the no-adapter ("original pre-unlearning model") case."""
    if not adapter_path:
        return fallback_base_model
    cfg_path = Path(adapter_path) / "adapter_config.json"
    if not cfg_path.exists():
        return fallback_base_model
    cfg = json.loads(cfg_path.read_text())
    return cfg.get("base_model_name_or_path", fallback_base_model)


def load_model(base_model, adapter_path, device):
    resolved_base = _adapter_base_model(adapter_path, base_model)
    if resolved_base != base_model and adapter_path:
        print(f"  NOTE: {adapter_path} was trained from base='{resolved_base}' "
              f"(not --base_model='{base_model}') -- merging onto its own recorded "
              f"base, per the Phase 2A Blocker-D fix.", flush=True)
    base = AutoModelForCausalLM.from_pretrained(resolved_base, torch_dtype=torch.bfloat16)
    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    else:
        model = base
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="microsoft/phi-1_5")
    p.add_argument("--adapter", required=True, help="Path to trained LoRA adapter directory (evaluated model)")
    p.add_argument("--oracle_adapter", required=True, help="Path to retain-only oracle LoRA adapter directory")
    p.add_argument("--split", default="forget05", choices=["forget01", "forget05", "forget10"])
    p.add_argument("--n_forget_probe", type=int, default=50,
                   help="Number of forget examples used for the KS-test (full set is 40/200/400; "
                        "probing a subset keeps official-metric evaluation tractable, matching the "
                        "n_probe convention already used for FQ_NLL/MU_ret in this codebase)")
    p.add_argument("--n_mu_probe", type=int, default=30)
    p.add_argument("--output_json", required=True)
    p.add_argument("--checkpoint_manifest", default=None,
                   help="Path to a checkpoint_manifests.json written by train_method(); "
                        "if given with --method, hard-fails unless --adapter's sha256 "
                        "matches that method's selected_checkpoint_sha256.")
    p.add_argument("--method", default=None, help="Method key to look up in --checkpoint_manifest")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU required")
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

    adapter_sha256 = _hash_adapter_dir(args.adapter)
    print(f"Evaluated-adapter sha256: {adapter_sha256}", flush=True)

    expected_selected_step = None
    if args.checkpoint_manifest:
        if not args.method:
            raise ValueError("--checkpoint_manifest requires --method")
        expected_selected_step = verify_checkpoint_identity(adapter_sha256, args.checkpoint_manifest, args.method)
        print(f"Checkpoint identity verified against manifest: selected_step={expected_selected_step}, "
              f"sha256 matches.", flush=True)

    tok = AutoTokenizer.from_pretrained(TOK_MODEL)
    tok.pad_token = tok.eos_token

    print(f"Loading official TOFU fields for split={args.split}...", flush=True)
    try:
        forget_perturbed = list(load_dataset("locuslab/TOFU", f"{args.split}_perturbed", split="train"))[:args.n_forget_probe]
        retain_perturbed = list(load_dataset("locuslab/TOFU", "retain_perturbed", split="train"))
        real_authors_perturbed = list(load_dataset("locuslab/TOFU", "real_authors_perturbed", split="train"))
        world_facts_perturbed = list(load_dataset("locuslab/TOFU", "world_facts_perturbed", split="train"))
    except Exception as e:
        raise RuntimeError(
            f"Failed to load required official TOFU dataset configs "
            f"({args.split}_perturbed / retain_perturbed / real_authors_perturbed / "
            f"world_facts_perturbed). Refusing to substitute diagnostic data. "
            f"Original error: {e}"
        )

    print("Loading evaluated model...", flush=True)
    model = load_model(args.base_model, args.adapter, device)
    print("Loading retain-only oracle...", flush=True)
    oracle = load_model(args.base_model, args.oracle_adapter, device)

    try:
        print(f"Computing official Forget Quality on {len(forget_perturbed)} forget examples...", flush=True)
        fq_official, r_model, r_oracle = compute_forget_quality_tofu(model, oracle, tok, forget_perturbed, device)
        print(f"  FQ_official (Truth-Ratio KS p-value) = {fq_official:.4f}", flush=True)

        print(f"Computing official Model Utility (n_probe={args.n_mu_probe} per source)...", flush=True)
        mu_official, per_source = compute_model_utility_tofu(
            model, tok, retain_perturbed, real_authors_perturbed, world_facts_perturbed,
            device, n_probe=args.n_mu_probe,
        )
        print(f"  MU_official (9-quantity harmonic mean) = {mu_official:.4f}", flush=True)
    except MissingTOFUFieldError as e:
        print(f"HARD FAIL: {e}", flush=True)
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"status": "failed_missing_fields", "error": str(e)}, open(args.output_json, "w"), indent=2)
        raise

    result = {
        "status": "ok",
        "adapter": args.adapter,
        "adapter_sha256": adapter_sha256,
        "selected_step": expected_selected_step,
        "checkpoint_identity_verified_against_manifest": bool(args.checkpoint_manifest),
        "oracle_adapter": args.oracle_adapter,
        "split": args.split,
        "n_forget_probe": len(forget_perturbed),
        "n_mu_probe": args.n_mu_probe,
        "fq_official": fq_official,
        "mu_official": mu_official,
        "mu_official_per_source": per_source,
        "note": "Independent reimplementation of TOFU Truth-Ratio FQ and 9-quantity MU; "
                "not verified byte-identical to the original authors' reference code "
                "(see forge/tofu_official.py docstring).",
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.output_json, "w"), indent=2)
    print(f"Saved -> {args.output_json}")


if __name__ == "__main__":
    main()
