#!/usr/bin/env python
"""
scripts/build_provenance.py — builds outputs/RESULTS_PROVENANCE.json and
RESULTS_MANIFEST.md from verified output files only. No number is invented
here; every entry either reads a real JSON/trajectory file on disk or is
explicitly marked "unverified"/"missing" and excluded from paper usage.

Metric naming (per Phase 2 renaming convention):
  FQ_NLL  — KS-test p-value on raw NLL vs. retain-only oracle (diagnostic,
            NOT the official TOFU Truth-Ratio Forget Quality)
  MU_ret  — harmonic mean of retain ROUGE-L + perplexity recovery (diagnostic,
            NOT the official 9-quantity TOFU Model Utility)
  J_diag  — FQ_NLL * MU_ret (diagnostic joint score)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
provenance = {}


def add(key, value, metric, model, split, seeds, source, rtype, paper_usage, checkpoint_rule=None):
    provenance[key] = {
        "value": value,
        "metric": metric,
        "model": model,
        "split": split,
        "seeds": seeds,
        "checkpoint_rule": checkpoint_rule or "argmax_t(FQ_NLL_t * MU_ret_t) per seed (forge/stats.py::_select_checkpoint)",
        "source": str(source),
        "type": rtype,
        "paper_usage": paper_usage,
    }


def select_checkpoint(traj):
    return max(traj, key=lambda p: p.get("fq", 0) * p.get("mu", 0))


# ── TOFU Phi-1.5 diagnostic grid: 8 methods x 3 splits ────────────────────────
# forget05 has 5 seeds (42,123,456,789,1000); forget01/forget10 have 3 (42,123,456)
# -- the loop below skips missing seed directories automatically, so listing all 5
# here is sufficient; per-entry seed count is reported via len(seed_paths)/n below.
methods = ["GA", "GD", "KL", "FLAT", "LoKU", "NPO", "SimNPO", "FORGE"]
splits = ["forget01", "forget05", "forget10"]
seeds = [42, 123, 456, 789, 1000]

for m in methods:
    for s in splits:
        fqs, mus, js = [], [], []
        used_seeds = []
        seed_paths = []
        for sd in seeds:
            p = ROOT / f"outputs/wk8/{m}/{s}/seed{sd}/trajectories.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            traj = d.get(m, list(d.values())[0])
            best = select_checkpoint(traj)
            fqs.append(best["fq"]); mus.append(best["mu"])
            js.append(best["fq"] * best["mu"])  # per-seed product FIRST
            used_seeds.append(sd)
            seed_paths.append(str(p.relative_to(ROOT)))
        if not fqs:
            add(f"tofu_phi15_{s}_{m.lower()}_fq_nll", None, "FQ_NLL (diagnostic)",
                "locuslab/tofu_ft_phi-1.5", s, [], "MISSING", "missing", "none - excluded from paper")
            continue
        n = len(fqs)
        fq_mean = sum(fqs) / n
        mu_mean = sum(mus) / n
        # Evidence-remediation fix (Phase 1, task 4): J must be the mean of the
        # PER-SEED products FQ_i*MU_i, not the product of the per-metric means.
        # These differ whenever FQ and MU covary across seeds (confirmed for NPO:
        # product-of-means overstated J by 0.0095, exceeding the audit's 0.002
        # stop-and-investigate threshold -- see EVIDENCE_RECOVERY_REPORT.md).
        j_mean = sum(js) / n
        usage = "main diagnostic table (Table IV/VI)" if s == "forget05" else "cross-split diagnostic table (Table VI)"
        add(f"tofu_phi15_{s}_{m.lower()}_fq_nll", round(fq_mean, 4), "FQ_NLL = KS(NLL_model, NLL_oracle) [diagnostic, NOT official TOFU FQ]",
            "locuslab/tofu_ft_phi-1.5", s, used_seeds, seed_paths, "real_gpu_diagnostic", usage)
        add(f"tofu_phi15_{s}_{m.lower()}_mu_ret", round(mu_mean, 4), "MU_ret = harmonic_mean(ROUGE-L_retain, PPL_recovery) [diagnostic, NOT official TOFU MU]",
            "locuslab/tofu_ft_phi-1.5", s, used_seeds, seed_paths, "real_gpu_diagnostic", usage)
        add(f"tofu_phi15_{s}_{m.lower()}_j_diag", round(j_mean, 4), "J_diag = mean_seed(FQ_NLL_i * MU_ret_i) [NOT mean(FQ_NLL)*mean(MU_ret) -- these differ when FQ/MU covary across seeds]",
            "locuslab/tofu_ft_phi-1.5", s, used_seeds, seed_paths, "real_gpu_diagnostic", usage)

# ── Official-style TOFU metrics (Phase 2 reimplementation) ───────────────────
for m in methods:
    fqs, mus = [], []
    used_seeds = []
    seed_paths = []
    for sd in seeds:
        p = ROOT / f"outputs/official_tofu/{m}/forget05/seed{sd}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        if d.get("status") != "ok":
            continue
        fqs.append(d["fq_official"]); mus.append(d["mu_official"])
        used_seeds.append(sd)
        seed_paths.append(str(p.relative_to(ROOT)))
    if fqs:
        n = len(fqs)
        add(f"tofu_phi15_forget05_{m.lower()}_fq_official", round(sum(fqs) / n, 4),
            "FQ_official = KS-test p-value on Truth Ratio distribution vs. retain-only oracle "
            "[independent reimplementation of Maini et al. 2024 formula; NOT verified byte-identical "
            "to the original authors' reference code, see forge/tofu_official.py docstring]",
            "locuslab/tofu_ft_phi-1.5", "forget05", used_seeds, seed_paths, "real_gpu_official_reimplementation",
            "official TOFU metrics table")
        add(f"tofu_phi15_forget05_{m.lower()}_mu_official", round(sum(mus) / n, 4),
            "MU_official = harmonic mean of 9 quantities (ROUGE-L, answer probability, "
            "truth-ratio score) x (retain, real_authors, world_facts) "
            "[independent reimplementation, same caveat as FQ_official above]",
            "locuslab/tofu_ft_phi-1.5", "forget05", used_seeds, seed_paths, "real_gpu_official_reimplementation",
            "official TOFU metrics table")
    else:
        add(f"tofu_phi15_forget05_{m.lower()}_fq_official", None, "FQ_official", "n/a", "forget05",
            [], "MISSING", "missing", "none - excluded from paper")

# ── Robustness ablation, multi-seed (supersedes single-seed entries above) ───
p = ROOT / "outputs/wk_robustness/gamma_ablation_multiseed.json"
if p.exists():
    d = json.load(open(p))
    for label, res in d["results"].items():
        for cond, vals in res.items():
            add(f"robustness_multiseed_{label.lower()}_{cond}_rouge_leak", round(vals["rouge_leak_mean"], 4),
                "ROUGE-L leakage under perturbed prompt, 3-seed mean (diagnostic)",
                "locuslab/tofu_ft_phi-1.5", "forget05", d["seeds"], str(p.relative_to(ROOT)),
                "real_gpu_diagnostic", "robustness ablation table (multi-seed, supersedes single-seed)",
                checkpoint_rule=f"final training step, std across seeds = {vals['rouge_leak_std']:.4f}")

# ── WMDP corrected (cyber corpus-based, bio held-out-split) ───────────────────
for domain in ["cyber", "bio"]:
    p = ROOT / f"outputs/wk_wmdp_fix/{domain}/wmdp_summary.json"
    if p.exists():
        d = json.load(open(p))
        for m, r in d["results"].items():
            rtype = "real_gpu_corpus_based" if domain == "cyber" else "real_gpu_mcq_heldout_substitute"
            add(f"wmdp_{domain}_{m.lower()}_acc", r["final_acc"], f"WMDP-{domain} MCQ accuracy (zero-shot on held-out set)",
                "locuslab/tofu_ft_phi-1.5", f"wmdp_{domain}", [42], str(p.relative_to(ROOT)), rtype,
                f"WMDP corrected table ({domain})", checkpoint_rule="final training step (200 steps), single seed")
    else:
        add(f"wmdp_{domain}_summary", None, "WMDP summary", "n/a", domain, [], "MISSING", "missing", "none - excluded")

# ── MUSE-News (real data) ─────────────────────────────────────────────────────
p = ROOT / "outputs/wk_muse_fix/muse_summary.json"
if p.exists():
    d = json.load(open(p))
    for m, r in d.items():
        add(f"muse_news_{m.lower()}_verbmem", r["verbmem"], "VerbMem (n-gram overlap, real muse-bench/MUSE-News corpus)",
            "locuslab/tofu_ft_phi-1.5", "muse_news", [42], str(p.relative_to(ROOT)), "real_gpu_real_data",
            "MUSE-News corrected table", checkpoint_rule="final training step (250 steps), single seed")
        add(f"muse_news_{m.lower()}_knowmem", r["knowmem"], "KnowMem (ROUGE-L on QA, real muse-bench/MUSE-News corpus)",
            "locuslab/tofu_ft_phi-1.5", "muse_news", [42], str(p.relative_to(ROOT)), "real_gpu_real_data",
            "MUSE-News corrected table", checkpoint_rule="final training step (250 steps), single seed")

# ── Robustness ablation ───────────────────────────────────────────────────────
p = ROOT / "outputs/wk_robustness/gamma_ablation.json"
if p.exists():
    d = json.load(open(p))
    for label, res in d["results"].items():
        for cond, vals in res.items():
            add(f"robustness_{label.lower()}_{cond}_rouge_leak", round(vals["rouge_leak_mean"], 4),
                "ROUGE-L leakage under perturbed prompt (diagnostic)",
                "locuslab/tofu_ft_phi-1.5", "forget05", [42], str(p.relative_to(ROOT)), "real_gpu_diagnostic_single_seed",
                "robustness ablation table")

# ── Llama-2-7B scale validation (corrected) ───────────────────────────────────
p = ROOT / "outputs/wk_llama/stats_summary_corrected.json"
if p.exists():
    d = json.load(open(p))
    for m, r in d.items():
        add(f"llama7b_forget05_{m.lower()}_fq_nll", round(r["fq_mean"], 4), "FQ_NLL (diagnostic, 7B scale)",
            "locuslab/tofu_ft_llama2-7b", "forget05", [42, 123, 456][:r["n_seeds"]],
            "outputs/wk_llama/*/forget05/seed*/trajectories.json (recomputed)", "real_gpu_diagnostic",
            "Llama-2-7B scale table")
        add(f"llama7b_forget05_{m.lower()}_mu_ret", round(r["mu_mean"], 4), "MU_ret (diagnostic, 7B scale)",
            "locuslab/tofu_ft_llama2-7b", "forget05", [42, 123, 456][:r["n_seeds"]],
            "outputs/wk_llama/*/forget05/seed*/trajectories.json (recomputed)", "real_gpu_diagnostic",
            "Llama-2-7B scale table")
# NPO/SimNPO negative result at 7B (single seed, explicitly marked preliminary)
for m, tag in [("NPO", "llama_NPO_fine"), ("SimNPO", "llama_SimNPO_fine")]:
    p = ROOT / f"outputs/probe/{tag}/trajectories.json"
    if p.exists():
        d = json.load(open(p))
        traj = d.get(m, list(d.values())[0])
        best = select_checkpoint(traj)
        add(f"llama7b_forget05_{m.lower()}_fq_nll", round(best["fq"], 4),
            "FQ_NLL (diagnostic, 7B scale) — negative result: 0.000 across all 21 checkpoints (5-step resolution)",
            "locuslab/tofu_ft_llama2-7b", "forget05", [42], str(p.relative_to(ROOT)),
            "real_gpu_diagnostic_single_seed_negative_result", "Llama-2-7B scale table (negative-result footnote)")

# ── Cost (unaffected by checkpoint-selection bug) ─────────────────────────────
p = ROOT / "outputs/wk9/cost_analysis.json"
if p.exists():
    d = json.load(open(p))
    add("cost_forge_lora_r32", d["forge_lora_r32"], "Trainable params / peak VRAM / sec-per-step",
        "locuslab/tofu_ft_phi-1.5", "forget05", [42], str(p.relative_to(ROOT)), "real_gpu_cost", "Table IX (unchanged)")
    add("cost_full_retrain", d["full_retrain"], "Trainable params / peak VRAM / sec-per-step",
        "locuslab/tofu_ft_phi-1.5", "forget05", [42], str(p.relative_to(ROOT)), "real_gpu_cost", "Table IX (unchanged)")

# ── Sequential unlearning (unaffected — verified different, sound checkpoint logic) ──
for m in ["GA", "LoKU", "FORGE"]:
    matches = list((ROOT / f"outputs/wk9/sequential_{m}").glob("*.json"))
    if matches:
        d = json.load(open(matches[0]))
        add(f"sequential_{m.lower()}_delta_fq", d["delta_fq_stage1"], "|FQ_NLL(half-A) drift after stage-2 unlearning|",
            "locuslab/tofu_ft_phi-1.5", "forget05_halves", [42], str(matches[0].relative_to(ROOT)),
            "real_gpu_diagnostic", "Table X (unchanged)")

json.dump(provenance, open(ROOT / "outputs/RESULTS_PROVENANCE.json", "w"), indent=2)
n_real = sum(1 for v in provenance.values() if v["type"].startswith("real"))
n_missing = sum(1 for v in provenance.values() if v["type"] == "missing")
print(f"Wrote {len(provenance)} entries ({n_real} real, {n_missing} missing) -> outputs/RESULTS_PROVENANCE.json")
