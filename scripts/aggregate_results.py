#!/usr/bin/env python
"""
scripts/aggregate_results.py — Phase 8 statistical reporting upgrade.

Produces, from real per-seed trajectory files only (outputs/wk8/):
  outputs/revision_stats/per_seed_all.csv       — every (method, split, seed) row
  outputs/revision_stats/summary_mean_std.csv   — mean, std, median per (method, split)
  outputs/revision_stats/bootstrap_ci.json      — 95% bootstrap CI per (method, split)
  outputs/revision_stats/effect_sizes.csv       — Cohen's d, FORGE vs each baseline
  figures/fig_pareto_dominance.pdf              — FQ_NLL x MU_ret Pareto scatter, forget05
  figures/data/pareto_dominance_forget05.csv    — the scatter's source data

Design choice per Phase 8 instructions: p-values are computed but explicitly
labeled secondary; mean±std/median/effect-size are the headline statistics,
since n=3 seeds gives implausibly overconfident p-values (df=2).
"""
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from scipy import stats as sc
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parent.parent
METHODS = ["GA", "GD", "KL", "FLAT", "LoKU", "NPO", "SimNPO", "FORGE"]
SPLITS = ["forget01", "forget05", "forget10"]
# forget05 has 5 seeds (42,123,456,789,1000); forget01/forget10 have 3 (42,123,456) --
# load_seed_values() skips missing seed dirs automatically, so listing all 5 is safe.
SEEDS = [42, 123, 456, 789, 1000]


def select_checkpoint(traj):
    return max(traj, key=lambda p: p.get("fq", 0) * p.get("mu", 0))


def load_seed_values(method, split):
    rows = []
    for seed in SEEDS:
        p = ROOT / f"outputs/wk8/{method}/{split}/seed{seed}/trajectories.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        traj = d.get(method, list(d.values())[0])
        best = select_checkpoint(traj)
        rows.append({"method": method, "split": split, "seed": seed,
                     "fq_nll": best["fq"], "mu_ret": best["mu"],
                     "j_diag": best["fq"] * best["mu"], "step": best["step"]})
    return rows


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return None
    pooled_std = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2) / (len(a) + len(b) - 2))
    if pooled_std == 0:
        return None
    return float((b.mean() - a.mean()) / pooled_std)


def bootstrap_ci(values, n=1000, conf=0.95):
    if len(values) == 0:
        return None, None, None
    boots = [np.mean(resample(values)) for _ in range(n)]
    lo = float(np.percentile(boots, (1 - conf) / 2 * 100))
    hi = float(np.percentile(boots, (1 + conf) / 2 * 100))
    return float(np.mean(values)), lo, hi


def main():
    out_dir = ROOT / "outputs/revision_stats"
    fig_dir = ROOT / "figures"
    fig_data_dir = fig_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_data_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for method in METHODS:
        for split in SPLITS:
            all_rows += load_seed_values(method, split)

    # per_seed_all.csv
    with open(out_dir / "per_seed_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "split", "seed", "fq_nll", "mu_ret", "j_diag", "step"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows -> {out_dir/'per_seed_all.csv'}")

    # summary_mean_std.csv + bootstrap_ci.json
    summary_rows = []
    ci_report = {}
    for method in METHODS:
        for split in SPLITS:
            rows = [r for r in all_rows if r["method"] == method and r["split"] == split]
            if not rows:
                continue
            for metric in ["fq_nll", "mu_ret", "j_diag"]:
                vals = [r[metric] for r in rows]
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0  # sample std (ddof=1), per Phase-1 remediation task 4: J_sd = sample_std(J_s)
                median = statistics.median(vals)
                summary_rows.append({"method": method, "split": split, "metric": metric,
                                     "mean": round(mean, 4), "std": round(std, 4),
                                     "median": round(median, 4), "n": len(vals),
                                     "values": ";".join(f"{v:.4f}" for v in vals)})
                ci_mean, ci_lo, ci_hi = bootstrap_ci(vals)
                ci_report.setdefault(split, {}).setdefault(metric, {})[method] = {
                    "mean": ci_mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": len(vals),
                }

    with open(out_dir / "summary_mean_std.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "split", "metric", "mean", "std", "median", "n", "values"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {len(summary_rows)} rows -> {out_dir/'summary_mean_std.csv'}")

    json.dump(ci_report, open(out_dir / "bootstrap_ci.json", "w"), indent=2)
    print(f"Wrote -> {out_dir/'bootstrap_ci.json'}")

    # effect_sizes.csv — FORGE vs each baseline, per split, per metric (secondary to p-values)
    effect_rows = []
    for split in SPLITS:
        forge_rows = [r for r in all_rows if r["method"] == "FORGE" and r["split"] == split]
        for method in METHODS:
            if method == "FORGE":
                continue
            base_rows = [r for r in all_rows if r["method"] == method and r["split"] == split]
            if not forge_rows or not base_rows:
                continue
            for metric in ["fq_nll", "mu_ret", "j_diag"]:
                a = [r[metric] for r in base_rows]
                b = [r[metric] for r in forge_rows]
                d = cohens_d(a, b)
                p = None
                if len(a) == len(b) and len(a) >= 2:
                    try:
                        _, p = sc.ttest_rel(b, a)
                    except Exception:
                        p = None
                effect_rows.append({"split": split, "baseline": method, "metric": metric,
                                    "cohens_d": round(d, 3) if d is not None else "",
                                    "p_value_secondary": round(float(p), 4) if p is not None else "",
                                    "note": "n=3: p-value is secondary/descriptive, see Phase 8/[R16]"})
    with open(out_dir / "effect_sizes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "baseline", "metric", "cohens_d", "p_value_secondary", "note"])
        w.writeheader()
        w.writerows(effect_rows)
    print(f"Wrote {len(effect_rows)} rows -> {out_dir/'effect_sizes.csv'}")

    # Pareto dominance figure, forget05
    scatter_rows = [r for r in all_rows if r["split"] == "forget05"]
    method_means = {}
    for method in METHODS:
        rows = [r for r in scatter_rows if r["method"] == method]
        if rows:
            method_means[method] = (statistics.mean(r["fq_nll"] for r in rows),
                                    statistics.mean(r["mu_ret"] for r in rows))

    with open(fig_data_dir / "pareto_dominance_forget05.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "fq_nll_mean", "mu_ret_mean"])
        for m, (fq, mu) in method_means.items():
            w.writerow([m, round(fq, 4), round(mu, 4)])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
        for m, (fq, mu) in method_means.items():
            ax.scatter(fq, mu, s=70, label=m)
            ax.annotate(m, (fq, mu), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_xlabel("FQ_NLL (diagnostic) ↑")
        ax.set_ylabel("MU_ret (diagnostic) ↑")
        ax.set_title("Forgetting–utility Pareto scatter, TOFU forget-5% (Phi-1.5)")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        plt.tight_layout()
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_dir / "fig_pareto_dominance.pdf")
        plt.close()
        print(f"Wrote -> {fig_dir/'fig_pareto_dominance.pdf'}")
    except ImportError:
        print("matplotlib not available — skipped figure, CSV data still written")


if __name__ == "__main__":
    main()
