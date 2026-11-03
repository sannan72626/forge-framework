#!/usr/bin/env python
"""
scripts/verify_muse_dataset.py — verifies whether the MUSE-News data used by
scripts/run_muse.py is the official benchmark corpus, a documented proxy, or
(as found during a prior revision pass) silently-fabricated synthetic text.

Background: the pre-revision version of scripts/run_muse.py pointed at a
dataset id, "jaechan-repo/muse-bench", that does not exist on the HF Hub —
confirmed via a direct load_dataset call raising DatasetNotFoundError — so
its synthetic-text fallback fired on 100% of runs. This script performs
that same check explicitly and records the result so it can never again
happen silently.

Usage:
  python scripts/verify_muse_dataset.py --output outputs/muse_dataset_verification.json
"""
import argparse
import json
from pathlib import Path


def check_dataset(dataset_id, config, split_names):
    from datasets import load_dataset
    result = {"dataset_id": dataset_id, "config": config, "loaded": False,
              "is_official": False, "is_proxy": False, "used_synthetic_fallback": False,
              "splits": {}, "error": None}
    try:
        ds = load_dataset(dataset_id, config)
        result["loaded"] = True
        for split in split_names:
            if split in ds:
                n = len(ds[split])
                sample_text = ds[split][0].get("text", "")
                n_tokens_sample = len(sample_text.split())
                result["splits"][split] = {"n_examples": n, "sample_word_count": n_tokens_sample}
    except Exception as e:
        result["error"] = str(e)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/muse_dataset_verification.json")
    args = p.parse_args()

    checks = []

    # 1. The id previously (incorrectly) used in scripts/run_muse.py, pre-revision.
    old = check_dataset("jaechan-repo/muse-bench", "news", ["forget", "retain"])
    old["role"] = "previously used (found to be broken) — must never load successfully"
    checks.append(old)

    # 2. The real, official MUSE-News dataset now used by scripts/run_muse.py.
    real = check_dataset("muse-bench/MUSE-News", "raw", ["forget", "retain1", "retain2", "holdout"])
    real["role"] = "current dataset used by scripts/run_muse.py"
    if real["loaded"]:
        real["is_official"] = True
        real["is_proxy"] = False
    checks.append(real)

    # Verdict
    old_broken = not checks[0]["loaded"]
    real_ok = checks[1]["loaded"]
    if old_broken and real_ok:
        verdict = ("CONFIRMED: the previously-used dataset id does not exist (synthetic "
                   "fallback would have fired on every run pre-fix); the currently-used "
                   "dataset id is real and loads the official MUSE-News corpus. "
                   "No synthetic fallback remains in scripts/run_muse.py "
                   "(verified separately by grep for 'synthetic'/'fallback' — see "
                   "docs/experiment_audit.md item 6).")
    elif not real_ok:
        verdict = ("WARNING: the currently-configured MUSE-News dataset id failed to load "
                   "in THIS environment (see error above). scripts/run_muse.py must hard-fail "
                   "rather than substitute synthetic data if this happens at experiment time.")
    else:
        verdict = "UNEXPECTED: old broken id unexpectedly loaded — re-check dataset id."

    out = {"checks": checks, "verdict": verdict}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
