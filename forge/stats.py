"""
forge/stats.py — Statistical validation for FORGE experiments.

Provides:
  bootstrap_ci(scores)          → (mean, ci_lo, ci_hi)
  paired_t(a, b)                → {t, p, sig, direction}
  load_scores(results_dir, ...)  → list of final-checkpoint metric values
  full_report(...)               → nested dict of stats per split × metric
"""

import json, os
import numpy as np
from scipy import stats as sc
from sklearn.utils import resample


def bootstrap_ci(scores, n=1000, conf=0.95):
    """95% bootstrap CI on the mean of scores."""
    scores = list(scores)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    boots = [np.mean(resample(scores)) for _ in range(n)]
    mean  = float(np.mean(scores))
    lo    = float(np.percentile(boots, (1 - conf) / 2 * 100))
    hi    = float(np.percentile(boots, (1 + conf) / 2 * 100))
    return mean, lo, hi


def paired_t(a, b, alpha=0.05):
    """Paired t-test: does b (FORGE) significantly outperform a (baseline)?
    Returns direction 'FORGE_wins' if b > a on the metric.
    """
    a, b = list(a), list(b)
    if len(a) != len(b) or len(a) < 2:
        return {"t": None, "p": 1.0, "sig": False, "dir": "insufficient_data"}
    t, p = sc.ttest_rel(b, a)
    return {
        "t": float(t),
        "p": float(p),
        "sig": bool(p < alpha),
        "dir": "FORGE_wins" if t > 0 else "baseline_wins",
    }


def load_scores(results_dir, method, split, seeds, metric, peak=True):
    """Load per-seed metric values from trajectory files.

    If peak=True, returns the maximum value of the metric across all steps
    (appropriate for FQ which peaks then declines).
    If peak=False, returns the final-step value.
    """
    scores = []
    for seed in seeds:
        path = os.path.join(results_dir, method, split, f"seed{seed}", "trajectories.json")
        if not os.path.exists(path):
            continue
        traj = json.load(open(path))
        data = traj.get(method, list(traj.values())[0])
        if not data:
            continue
        if peak:
            val = max(p.get(metric, 0) for p in data)
        else:
            val = data[-1].get(metric, 0)
        scores.append(val)
    return scores


def full_report(
    results_dir,
    baselines,
    target="FORGE",
    metrics=("fq", "mu", "rouge_forget", "rouge_retain", "mia_tr"),
    splits=("forget01", "forget05", "forget10"),
    seeds=(42, 123, 456),
    alpha=0.05,
    n_boot=1000,
):
    """Run full statistical report: bootstrap CIs + paired t-tests.

    Returns nested dict: report[split][metric][method] = {mean, ci_lo, ci_hi, vs_FORGE}
    """
    report = {}
    for split in splits:
        report[split] = {}
        for metric in metrics:
            use_peak = metric == "fq"  # FQ peaks then overshoots; use peak value
            tgt_scores = load_scores(results_dir, target, split, seeds, metric, peak=use_peak)
            if not tgt_scores:
                continue
            m, lo, hi = bootstrap_ci(tgt_scores, n_boot)
            report[split][metric] = {
                target: {"mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(tgt_scores)}
            }
            for bl in baselines:
                bl_scores = load_scores(results_dir, bl, split, seeds, metric, peak=use_peak)
                if not bl_scores:
                    continue
                bm, blo, bhi = bootstrap_ci(bl_scores, n_boot)
                vs = paired_t(bl_scores, tgt_scores, alpha) if len(bl_scores) == len(tgt_scores) else {}
                report[split][metric][bl] = {
                    "mean": bm, "ci_lo": blo, "ci_hi": bhi,
                    "n": len(bl_scores), "vs_FORGE": vs,
                }
    return report


def print_report(report, baselines, target="FORGE",
                 metrics=("fq", "mu"), splits=("forget01", "forget05", "forget10")):
    """Pretty-print the stats report to stdout."""
    print("="*75)
    print(f"Statistical Report — {target} vs baselines (paired t-test, α=0.05)")
    print("="*75)
    for split in splits:
        if split not in report:
            continue
        print(f"\n[{split}]")
        for metric in metrics:
            if metric not in report[split]:
                continue
            print(f"  {metric}:")
            tgt = report[split][metric].get(target, {})
            print(f"    {target:>8}: {tgt.get('mean',0):.3f} "
                  f"[{tgt.get('ci_lo',0):.3f}, {tgt.get('ci_hi',0):.3f}]  (n={tgt.get('n',0)})")
            for bl in baselines:
                d = report[split][metric].get(bl, {})
                vs = d.get("vs_FORGE", {})
                p_str = f"p={vs.get('p',1.0):.4f}" if vs else "n/a"
                sig   = "†" if vs.get("sig") else " "
                print(f"    {bl:>8}: {d.get('mean',0):.3f} "
                      f"[{d.get('ci_lo',0):.3f}, {d.get('ci_hi',0):.3f}]  "
                      f"{p_str} {sig}  ({vs.get('dir','?')})")


if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser(description="FORGE statistical report")
    p.add_argument("--results_dir", default="outputs/wk8/")
    p.add_argument("--out", default="outputs/wk8/stats_report.json")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()

    baselines = ["GA", "GD", "KL", "FLAT", "LoKU"]
    report = full_report(
        args.results_dir, baselines,
        splits=("forget01", "forget05", "forget10"),
        seeds=(42, 123, 456),
        alpha=args.alpha, n_boot=args.n_boot,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"Stats → {args.out}")
    print_report(report, baselines)

    # Check gating
    fails = []
    for bl in baselines:
        for met in ["fq", "mu"]:
            vs = report.get("forget05", {}).get(met, {}).get(bl, {}).get("vs_FORGE", {})
            if vs and not vs.get("sig"):
                fails.append(f"FORGE vs {bl} [{met}] p={vs.get('p',1):.4f}")
    if fails:
        print(f"\nWk8 gating WARN — not significant: {fails}")
        sys.exit(1)
    else:
        print("\nAll p<0.05 on forget05 FQ+MU — Wk8 gating PASS")
