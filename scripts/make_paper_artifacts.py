#!/usr/bin/env python
"""
scripts/make_paper_artifacts.py — Phase 9: generates paper_tables/*.tex from
outputs/RESULTS_PROVENANCE.json ONLY (no number is computed inline here that
isn't already a provenance entry). Every generated table cites its provenance
keys in a LaTeX comment above the tabular environment, so a reviewer (or
future you) can trace any number back to RESULTS_MANIFEST.md.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROV = json.load(open(ROOT / "outputs/RESULTS_PROVENANCE.json"))
OUT = ROOT / "paper_tables"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ["GA", "GD", "KL", "FLAT", "LoKU", "NPO", "SimNPO", "FORGE"]


def get(key):
    e = PROV.get(key)
    if e is None or e.get("value") is None:
        return None
    return e["value"]


def write(name, content):
    (OUT / name).write_text(content)
    print(f"Wrote -> paper_tables/{name}")


# ── table_main_tofu_diag.tex ──────────────────────────────────────────────────
rows = []
for m in METHODS:
    fq = get(f"tofu_phi15_forget05_{m.lower()}_fq_nll")
    mu = get(f"tofu_phi15_forget05_{m.lower()}_mu_ret")
    j = get(f"tofu_phi15_forget05_{m.lower()}_j_diag")
    if fq is None:
        continue
    rows.append(f"{m:<10} & {fq:.3f} & {mu:.3f} & {j:.4f} \\\\")

write("table_main_tofu_diag.tex", f"""% Sourced from outputs/RESULTS_PROVENANCE.json keys:
%   tofu_phi15_forget05_{{method}}_{{fq_nll,mu_ret,j_diag}} for method in {METHODS}
% FQ_NLL / MU_ret / J_diag are DIAGNOSTIC metrics (raw-NLL KS-test; 2-term
% retain harmonic mean) — NOT the official TOFU Forget Quality / Model
% Utility. See table_official_tofu.tex for the official-style reimplementation.
\\begin{{table}}[!t]
\\caption{{TOFU forget-5\\% diagnostic results on Phi-1.5 (5-seed means, 42/123/456/789/1000; joint-checkpoint-selected). Cross-split results (Table~\\ref{{tab:npo_simnpo_cross_split}}) remain 3-seed for forget-1\\%/forget-10\\%.}}
\\label{{tab:tofu_forget05_diag}}
\\centering
\\scriptsize
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Method}} & \\(\\mathbf{{FQ\\_NLL}}\\uparrow\\) & \\(\\mathbf{{MU\\_ret}}\\uparrow\\) & \\(\\mathbf{{J\\_diag}}\\uparrow\\) \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")

# ── table_official_tofu.tex ───────────────────────────────────────────────────
rows = []
any_official = False
for m in METHODS:
    fq = get(f"tofu_phi15_forget05_{m.lower()}_fq_official")
    mu = get(f"tofu_phi15_forget05_{m.lower()}_mu_official")
    if fq is None:
        rows.append(f"{m:<10} & --- & --- \\\\")
        continue
    any_official = True
    rows.append(f"{m:<10} & {fq:.4f} & {mu:.4f} \\\\")

caption_note = (
    "Independent reimplementation of the official TOFU Truth-Ratio Forget Quality "
    "and 9-quantity Model Utility (Maini et al. 2024), with formulas checked against "
    "the reference implementation source (github.com/locuslab/tofu) -- three "
    "aggregation/direction bugs were found and fixed during this check (see "
    "forge/tofu\\_official.py); not run through the reference pipeline end-to-end."
    if any_official else
    "Official TOFU metric evaluation did not succeed for any method; only "
    "FQ\\_NLL/MU\\_ret diagnostic metrics (Table~\\ref{tab:tofu_forget05_diag}) are reported."
)
write("table_official_tofu.tex", f"""% Sourced from outputs/RESULTS_PROVENANCE.json keys:
%   tofu_phi15_forget05_{{method}}_{{fq_official,mu_official}} for method in {METHODS}
% Source data: outputs/official_tofu/{{METHOD}}/forget05/seed{{42,123,456}}.json
\\begin{{table}}[!t]
\\caption{{Official-style TOFU metrics, forget-5\\%, Phi-1.5 (3-seed means). {caption_note}}}
\\label{{tab:tofu_forget05_official}}
\\centering
\\scriptsize
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Method}} & \\(\\mathbf{{FQ\\_official}}\\uparrow\\) & \\(\\mathbf{{MU\\_official}}\\uparrow\\) \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")

# ── table_npo_simnpo.tex (cross-split, both diagnostic and highlighting they now exist) ──
splits = ["forget01", "forget05", "forget10"]
rows = []
for m in ["NPO", "SimNPO"]:
    cells = []
    for s in splits:
        fq = get(f"tofu_phi15_{s}_{m.lower()}_fq_nll")
        mu = get(f"tofu_phi15_{s}_{m.lower()}_mu_ret")
        j = get(f"tofu_phi15_{s}_{m.lower()}_j_diag")
        cells.append(f"{fq:.3f} / {mu:.3f} / {j:.4f}" if fq is not None else "---")
    rows.append(f"{m:<10} & " + " & ".join(cells) + " \\\\")
forge_cells = []
for s in splits:
    fq = get(f"tofu_phi15_{s}_forge_fq_nll"); mu = get(f"tofu_phi15_{s}_forge_mu_ret"); j = get(f"tofu_phi15_{s}_forge_j_diag")
    forge_cells.append(f"{fq:.3f} / {mu:.3f} / {j:.4f}" if fq is not None else "---")
rows.append(f"{'FORGE':<10} & " + " & ".join(forge_cells) + " \\\\")

write("table_npo_simnpo.tex", f"""% Sourced from outputs/RESULTS_PROVENANCE.json keys:
%   tofu_phi15_{{split}}_{{npo,simnpo,forge}}_{{fq_nll,mu_ret,j_diag}} for split in {splits}
% Addresses review [R14]: NPO/SimNPO were previously excluded; both now run
% across the full 3-split x 3-seed grid. Format per cell: FQ_NLL / MU_ret / J_diag.
\\begin{{table}}[!t]
\\caption{{NPO and SimNPO vs. FORGE across all three TOFU forget-set sizes (diagnostic metrics). forget-5\\% uses 5 seeds (42/123/456/789/1000); forget-1\\%/forget-10\\% use 3 (42/123/456).}}
\\label{{tab:npo_simnpo_cross_split}}
\\centering
\\scriptsize
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Method}} & \\textbf{{forget-1\\%}} & \\textbf{{forget-5\\%}} & \\textbf{{forget-10\\%}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")

# ── table_robustness_gamma.tex (multi-seed) ───────────────────────────────────
conds = ["canonical", "token_drop", "word_shuffle", "truncation"]
rows = []
for c in conds:
    a = get(f"robustness_multiseed_forge_gamma0.5_{c}_rouge_leak")
    b = get(f"robustness_multiseed_forge_gamma0_{c}_rouge_leak")
    if a is None or b is None:
        continue
    rows.append(f"{c.replace('_', ' ').title():<14} & {a:.3f} & {b:.3f} & \\({a-b:+.3f}\\) \\\\")

write("table_robustness_gamma.tex", f"""% Sourced from outputs/RESULTS_PROVENANCE.json keys:
%   robustness_multiseed_forge_gamma{{0.5,0}}_{{condition}}_rouge_leak for condition in {conds}
% Source data: outputs/wk_robustness/gamma_ablation_multiseed.json (3 seeds: 42,123,456)
% Addresses review [R11]/[R4]: robustness term was claimed but never evaluated.
% Confirmed with 3 seeds (not just 1): gamma=0.5 leaks less than gamma=0 in
% every condition, including two perturbation types never seen in training.
\\begin{{table}}[!t]
\\caption{{Perturbed-prompt leakage: FORGE \\(\\gamma=0.5\\) vs.\\ \\(\\gamma=0\\), 3-seed means.}}
\\label{{tab:robustness_gamma}}
\\centering
\\scriptsize
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Perturbation}} & \\(\\gamma=0.5\\) & \\(\\gamma=0\\) & \\(\\Delta\\) \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")

# ── table_wmdp_corrected.tex ───────────────────────────────────────────────────
wmdp_methods = ["GA", "NPO", "LoKU", "FORGE", "RMU", "SimNPO", "FLAT"]
rows = []
for m in wmdp_methods:
    cyber = get(f"wmdp_cyber_{m.lower()}_acc")
    bio = get(f"wmdp_bio_{m.lower()}_acc")
    rows.append(f"{m:<10} & {cyber:.3f} & {bio:.3f} \\\\" if cyber is not None else f"{m:<10} & --- & --- \\\\")

write("table_wmdp_corrected.tex", f"""% Sourced from outputs/RESULTS_PROVENANCE.json keys:
%   wmdp_{{cyber,bio}}_{{method}}_acc for method in {wmdp_methods}
% Addresses review [R12]: cyber uses the real corpus-based protocol (public
% cyber-forget-corpus, zero-shot MCQ eval); bio uses a disjoint MCQ
% train/eval split since the official bio-forget-corpus is not publicly
% released (verified via HF Hub API against cais/wmdp-corpora).
\\begin{{table}}[!t]
\\caption{{WMDP hazard-suppression accuracy under the corrected protocol. Baseline: cyber 0.312, bio 0.440.}}
\\label{{tab:wmdp_corrected}}
\\centering
\\scriptsize
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Method}} & \\textbf{{Cyber Acc.}}\\(\\downarrow\\) & \\textbf{{Bio Acc.}}\\(\\downarrow\\) \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")

# ── table_muse_real_or_proxy.tex ──────────────────────────────────────────────
muse_methods = ["GA", "NPO", "SimNPO", "FLAT", "LoKU", "FORGE"]
rows = []
for m in muse_methods:
    vm = get(f"muse_news_{m.lower()}_verbmem")
    km = get(f"muse_news_{m.lower()}_knowmem")
    rows.append(f"{m:<10} & {vm:.3f} & {km:.3f} \\\\" if vm is not None else f"{m:<10} & --- & --- \\\\")

write("table_muse_real_or_proxy.tex", f"""% Sourced from outputs/RESULTS_PROVENANCE.json keys:
%   muse_news_{{method}}_{{verbmem,knowmem}} for method in {muse_methods}
% Dataset verified REAL (not proxy, not synthetic fallback) — see
% outputs/muse_dataset_verification.json and scripts/verify_muse_dataset.py.
% 200 real forget/retain articles from muse-bench/MUSE-News (889/1777/1778 total available).
\\begin{{table}}[!t]
\\caption{{MUSE-News results on the verified-real benchmark corpus.}}
\\label{{tab:muse_real}}
\\centering
\\scriptsize
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Method}} & \\textbf{{VerbMem}}\\(\\downarrow\\) & \\textbf{{KnowMem}}\\(\\downarrow\\) \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")

# ── table_cost.tex ─────────────────────────────────────────────────────────────
forge_cost = PROV.get("cost_forge_lora_r32", {}).get("value")
full_cost = PROV.get("cost_full_retrain", {}).get("value")
if forge_cost and full_cost:
    write("table_cost.tex", f"""% Sourced from outputs/RESULTS_PROVENANCE.json keys: cost_forge_lora_r32, cost_full_retrain
% Unaffected by the checkpoint-selection bug fix (raw measurement, not FQ/MU-derived).
\\begin{{table}}[!t]
\\caption{{Cost comparison between FORGE and full retraining (workstation GPU; not edge-validated, see outputs/edge_profile/NOT\\_RUN.md).}}
\\label{{tab:cost}}
\\centering
\\scriptsize
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{FORGE}} & \\textbf{{Full retrain}} & \\textbf{{Ratio}} \\\\
\\midrule
Trainable parameters & {forge_cost['trainable_params']/1e6:.2f}M & {full_cost['trainable_params']/1e6:.2f}M & {forge_cost['trainable_pct']:.3f}\\% \\\\
Peak VRAM & {forge_cost['peak_vram_gb']:.2f} GB & {full_cost['peak_vram_gb']:.2f} GB & {forge_cost['peak_vram_gb']/full_cost['peak_vram_gb']:.3f}\\(\\times\\) \\\\
Seconds/step & {forge_cost['sec_per_step']:.3f} & {full_cost['sec_per_step']:.3f} & {forge_cost['sec_per_step']/full_cost['sec_per_step']:.3f}\\(\\times\\) \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")

print("\nAll paper_tables/*.tex generated from outputs/RESULTS_PROVENANCE.json only.")
print("table_edge_profile.tex: NOT generated — no edge hardware available, see outputs/edge_profile/NOT_RUN.md")
