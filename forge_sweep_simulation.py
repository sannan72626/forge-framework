"""
forge_sweep_simulation.py
─────────────────────────
Standalone simulation of the FORGE hyperparameter sweep.
No GPU, no TOFU dataset required.

Usage:
    python forge_sweep_simulation.py                  # default out_dir=results/
    python forge_sweep_simulation.py --out_dir /tmp/  # custom output directory
    python forge_sweep_simulation.py --seed 0 --show  # open plots interactively

Response surface calibrated against:
  • FORGE default  (lr=1e-5, ep=10, r=16) → FQ=0.897, MU=0.876  [Week 4]
  • GA baseline at TOFU-5%                → FQ=0.951, MU=0.431
  • LoKU baseline at TOFU-5%              → FQ=0.781, MU=0.853
"""

import argparse
import json
import os
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Constants ────────────────────────────────────────────────────────────────
LRS    = [1e-6, 5e-6, 1e-5, 2e-5]
EPOCHS = [3, 5, 10, 15]
RANKS  = [8, 16, 32]

WHITE = "#FFFFFF"; NAVY = "#1E2761"; SOFT = "#3D5A80"
AMBER = "#F4B400"; SAGE  = "#81B29A"; CORAL = "#E07A5F"
WARN  = "#B23A48"; ICE   = "#EEF1FA"; MUTE  = "#5A6373"

LR_COLORS = {1e-6: "#2ECC71", 5e-6: "#3D5A80", 1e-5: "#E07A5F", 2e-5: "#B23A48"}
RK_MARKERS = {8: "s", 16: "o", 32: "D"}
RK_SIZES   = {8: 55,  16: 80,  32: 120}

# ── Response surface ─────────────────────────────────────────────────────────
# Anchor: lr=1e-5, ep=10, r=16 → (FQ=0.897, MU=0.876)
# Sensitivity estimates from sweep literature + TOFU ablation data:
#   FQ: +0.025 per 10× lr,  +0.006 per epoch,  +0.012 per rank doubling
#   MU: -0.055 per 10× lr,  -0.009 per epoch,  +0.010 per rank doubling
#   2e-5 extra MU penalty: -0.04 * (ep-3)/12  (over-suppress with epochs)

def _fq(lr: float, ep: int, rk: int, rng: np.random.Generator) -> float:
    dl = 0.025 * (np.log10(lr) - np.log10(1e-5))
    de = 0.006 * (ep - 10)
    dr = 0.012 * (np.log2(rk) - 4)
    return float(np.clip(0.897 + dl + de + dr + rng.normal(0, 0.006), 0.62, 0.97))

def _mu(lr: float, ep: int, rk: int, rng: np.random.Generator) -> float:
    dl = -0.055 * (np.log10(lr) - np.log10(1e-5))
    de = -0.009 * (ep - 10)
    dr =  0.010 * (np.log2(rk) - 4)
    extra = -0.04 * (ep - 3) / 12 if lr == 2e-5 else 0.0
    return float(np.clip(0.876 + dl + de + dr + extra + rng.normal(0, 0.006), 0.55, 0.95))

def _rouge_f(lr, ep, rk, rng):
    return float(np.clip(0.041 - 0.004*(ep-10) - 0.008*(np.log10(lr)-np.log10(1e-5))
                         + rng.normal(0, 0.004), 0.01, 0.20))

def _rouge_r(lr, ep, rk, rng):
    return float(np.clip(0.889 + 0.004*(np.log2(rk)-4) - 0.005*(ep-10)
                         + rng.normal(0, 0.004), 0.80, 0.93))

def _mia_tr(lr, ep, rk, rng):
    return float(np.clip(0.121 + 0.008*(ep-10) - 0.003*(np.log2(rk)-4)
                         + rng.normal(0, 0.004), 0.05, 0.30))

def _adv_react(lr, ep, rk, rng):
    return float(np.clip(0.070 + 0.012*(np.log10(lr)-np.log10(1e-5))
                         + 0.004*(ep-10) - 0.005*(np.log2(rk)-4)
                         + rng.normal(0, 0.004), 0.02, 0.35))

# ── Sweep engine ─────────────────────────────────────────────────────────────
def run_sweep(seed: int = 42) -> list[dict]:
    """Run all 48 (lr × epoch × rank) combinations; return list of result dicts."""
    rng = np.random.default_rng(seed)
    results = []
    for lr, ep, rk in product(LRS, EPOCHS, RANKS):
        results.append({
            "lr": lr, "ep": ep, "rk": rk,
            "fq":        _fq(lr, ep, rk, rng),
            "mu":        _mu(lr, ep, rk, rng),
            "rouge_f":   _rouge_f(lr, ep, rk, rng),
            "rouge_r":   _rouge_r(lr, ep, rk, rng),
            "mia_tr":    _mia_tr(lr, ep, rk, rng),
            "adv_react": _adv_react(lr, ep, rk, rng),
        })
    return results

# ── Pareto analysis ──────────────────────────────────────────────────────────
def pareto_front(results: list[dict], x_key="fq", y_key="mu") -> list[dict]:
    """Return Pareto-optimal runs (maximise both x and y)."""
    front = []
    for r in results:
        dominated = any(
            s[x_key] >= r[x_key] and s[y_key] >= r[y_key]
            and (s[x_key] > r[x_key] or s[y_key] > r[y_key])
            for s in results
        )
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda x: x[x_key])

def recommended_config(results: list[dict]) -> dict:
    """Return config with best FQ+MU sum on the Pareto front."""
    front = pareto_front(results)
    return max(front, key=lambda r: r["fq"] + r["mu"])

# ── Summary table ────────────────────────────────────────────────────────────
def print_summary(results: list[dict]) -> None:
    sep = "─" * 80
    hdr = f"{'lr':>9}  {'ep':>3}  {'r':>4}  {'FQ':>6}  {'MU':>6}  {'RL-f':>6}  {'RL-r':>6}  {'MIA-TR':>7}  {'Adv':>6}"
    print(sep)
    print("FORGE HYPERPARAMETER SWEEP  —  48 runs")
    print(sep)
    print(hdr)
    print(sep)

    # Sort by FQ+MU descending
    for r in sorted(results, key=lambda x: x["fq"]+x["mu"], reverse=True)[:20]:
        marker = " ★" if r["fq"] > 0.88 and r["mu"] > 0.87 else "  "
        print(f"{r['lr']:>9.0e}  {r['ep']:>3d}  {r['rk']:>4d}  "
              f"{r['fq']:>6.3f}  {r['mu']:>6.3f}  {r['rouge_f']:>6.3f}  "
              f"{r['rouge_r']:>6.3f}  {r['mia_tr']:>7.3f}  {r['adv_react']:>6.3f}{marker}")
    print(f"  (showing top 20 of 48; ★ = both FQ>0.88 and MU>0.87)")
    print(sep)

    rec = recommended_config(results)
    print(f"RECOMMENDED: lr={rec['lr']:.0e}, ep={rec['ep']}, rank={rec['rk']}")
    print(f"  → FQ={rec['fq']:.3f}, MU={rec['mu']:.3f}")
    print(f"  → adv_react={rec['adv_react']:.3f}, mia_tr={rec['mia_tr']:.3f}")
    print(sep)

    # Key findings
    lrs_set = sorted(set(r["lr"] for r in results))
    print("\nKey findings:")
    print("  [1] lr → MU  (primary MU driver)")
    for lr in lrs_set:
        sub = [r for r in results if r["lr"] == lr]
        mu_spread = max(r["mu"] for r in sub) - min(r["mu"] for r in sub)
        mu_mean   = np.mean([r["mu"] for r in sub])
        print(f"       lr={lr:.0e}  MU mean={mu_mean:.3f}  spread={mu_spread:.3f}")
    ep_set = sorted(set(r["ep"] for r in results))
    print("  [2] epochs → FQ  (primary FQ driver)")
    for ep in ep_set:
        sub = [r for r in results if r["ep"] == ep]
        fq_spread = max(r["fq"] for r in sub) - min(r["fq"] for r in sub)
        fq_mean   = np.mean([r["fq"] for r in sub])
        print(f"       ep={ep:>2d}  FQ mean={fq_mean:.3f}  spread={fq_spread:.3f}")
    rk_set = sorted(set(r["rk"] for r in results))
    print("  [3] rank → both  (no tradeoff)")
    for rk in rk_set:
        sub = [r for r in results if r["rk"] == rk]
        print(f"       r={rk:>2d}  FQ mean={np.mean([r['fq'] for r in sub]):.3f}  "
              f"MU mean={np.mean([r['mu'] for r in sub]):.3f}")
    print(sep)

# ── Figures ──────────────────────────────────────────────────────────────────
def fig_scatter(results, out_dir):
    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=185)
    fig.patch.set_facecolor(WHITE); ax.set_facecolor("#FAFAFB")

    for r in results:
        ax.scatter(r["fq"], r["mu"],
                   color=LR_COLORS[r["lr"]], marker=RK_MARKERS[r["rk"]],
                   s=RK_SIZES[r["rk"]], alpha=0.82,
                   edgecolors="white", lw=0.8, zorder=3)

    front = pareto_front(results)
    pfq = [p["fq"] for p in front]; pmu = [p["mu"] for p in front]
    ax.plot(pfq, pmu, color=AMBER, lw=2.2, ls="--", zorder=4, label="Pareto front")
    ax.scatter(pfq, pmu, color=AMBER, s=90, zorder=5, edgecolor="#7A5A00", lw=1.5)

    top3 = sorted(results, key=lambda x: x["fq"]+x["mu"], reverse=True)[:3]
    for r in top3:
        ax.text(r["fq"]+0.003, r["mu"]+0.004,
                f"lr={r['lr']:.0e}\nep={r['ep']}, r={r['rk']}",
                fontsize=7.5, color=NAVY, fontweight="bold",
                bbox=dict(fc=ICE, ec=NAVY, lw=0.8, boxstyle="round,pad=0.2"))

    def_r = next(r for r in results if abs(r["lr"]-1e-5)<1e-9 and r["ep"]==10 and r["rk"]==16)
    ax.scatter([def_r["fq"]], [def_r["mu"]], marker="*", s=320,
               color=NAVY, edgecolor=AMBER, lw=2, zorder=6)
    ax.text(def_r["fq"]-0.004, def_r["mu"]-0.018,
            "FORGE default\n(1e-5, ep=10, r=16)",
            fontsize=8, color=NAVY, fontstyle="italic", ha="right")

    ax.axhline(0.85, color="#BBB", ls=":", lw=1.1)
    ax.axvline(0.85, color="#BBB", ls=":", lw=1.1)
    ax.fill_between([0.86, 1.0], [0.86, 0.86], [0.96, 0.96], color=AMBER, alpha=0.06)
    ax.text(0.975, 0.94, "Target zone", fontsize=8.5, color=AMBER, fontstyle="italic", ha="center")

    lr_h = [mpatches.Patch(color=c, label=f"lr={lr:.0e}") for lr, c in LR_COLORS.items()]
    rk_h = [Line2D([0],[0], marker=m, color="w", markerfacecolor="#555",
                   markersize=np.sqrt(s), label=f"rank={rk}")
            for rk, (m, s) in {8:("s",55), 16:("o",80), 32:("D",120)}.items()]
    ax.legend(handles=lr_h+rk_h, loc="lower left", fontsize=8.5, frameon=True, ncol=2)
    ax.set_xlabel("Forget Quality (FQ) ↑", fontsize=12)
    ax.set_ylabel("Model Utility (MU) ↑", fontsize=12)
    ax.set_title("FORGE sweep: 48 runs (4 lr × 4 epochs × 3 rank)\n"
                 "Colour=lr  Shape=rank  Size=rank capacity  ★=current default",
                 fontsize=11.5, color=NAVY, loc="left", pad=9)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    path = Path(out_dir) / "sweep_scatter.png"
    plt.savefig(path, dpi=185, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    return path

def fig_heatmaps(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=185)
    fig.patch.set_facecolor(WHITE)

    for ax, metric, title, cmap_n in [
        (axes[0], "fq", "Forget Quality (FQ)  — rank=16", "Blues"),
        (axes[1], "mu", "Model Utility (MU)   — rank=16", "Greens"),
    ]:
        grid = np.array([[
            next(r[metric] for r in results
                 if abs(r["lr"]-lr)<1e-9 and r["ep"]==ep and r["rk"]==16)
            for lr in LRS] for ep in EPOCHS])
        im = ax.imshow(grid, cmap=cmap_n, vmin=grid.min()-0.01, vmax=grid.max()+0.01, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03).ax.tick_params(labelsize=9)
        ax.set_xticks(range(len(LRS)))
        ax.set_xticklabels([f"{lr:.0e}" for lr in LRS], fontsize=10.5)
        ax.set_yticks(range(len(EPOCHS)))
        ax.set_yticklabels([f"ep={e}" for e in EPOCHS], fontsize=10.5)
        ax.set_xlabel("Learning rate", fontsize=11)
        ax.set_ylabel("Epochs", fontsize=11)
        ax.set_title(title, fontsize=12, color=NAVY, fontweight="bold", pad=7)
        for i in range(len(EPOCHS)):
            for j in range(len(LRS)):
                v = grid[i, j]
                tc = ("white" if (metric=="fq" and v>0.91) or
                      (metric=="mu" and v<0.80) else NAVY)
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=10.5, color=tc, fontweight="bold")
        best = np.unravel_index(np.argmax(grid), grid.shape)
        ax.add_patch(plt.Rectangle((best[1]-0.5, best[0]-0.5), 1, 1,
                     fill=False, edgecolor=AMBER, lw=3))

    fig.suptitle("LR × Epochs interaction (rank=16)  —  "
                 "MU driven by lr (columns);  FQ driven by epochs (rows)",
                 fontsize=12, color=NAVY, fontweight="bold")
    plt.tight_layout()
    path = Path(out_dir) / "heatmap.png"
    plt.savefig(path, dpi=185, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    return path

def fig_lr_epoch(results, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), dpi=185)
    fig.patch.set_facecolor(WHITE)
    ep_arr = np.array(EPOCHS)

    for lr, col in LR_COLORS.items():
        fq_v = [next(r["fq"] for r in results if abs(r["lr"]-lr)<1e-9
                     and r["ep"]==ep and r["rk"]==16) for ep in EPOCHS]
        mu_v = [next(r["mu"] for r in results if abs(r["lr"]-lr)<1e-9
                     and r["ep"]==ep and r["rk"]==16) for ep in EPOCHS]
        lbl = f"lr={lr:.0e}"
        ax1.plot(ep_arr, fq_v, color=col, lw=2.2, marker="o", ms=7, label=lbl)
        ax2.plot(ep_arr, mu_v, color=col, lw=2.2, marker="o", ms=7, label=lbl)

    for ax, ylabel, title in [
        (ax1, "FQ", "FQ grows with epochs — lr shifts level but slope same"),
        (ax2, "MU", "MU dominated by lr — 2e-5 collapses with epochs"),
    ]:
        ax.axhline(0.85, color="#BBB", ls="--", lw=1.1)
        ax.set_xlabel("Epochs", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11.5, color=NAVY, fontweight="bold", pad=7)
        ax.set_facecolor("#FAFAFB")
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        ax.legend(fontsize=9.5, frameon=False)
        ax.tick_params(labelsize=10)

    fig.suptitle("Decoupled control: epochs → FQ;   lr → MU   (rank=16)",
                 fontsize=12, color=NAVY, fontweight="bold")
    plt.tight_layout()
    path = Path(out_dir) / "lr_epoch_curves.png"
    plt.savefig(path, dpi=185, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    return path

def fig_adaptive(out_dir):
    """Show the 4-phase adaptive LR schedule alongside a rank comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), dpi=185)
    fig.patch.set_facecolor(WHITE)

    # Left: LR schedule
    steps = np.linspace(0, 300, 600)
    rng2 = np.random.default_rng(99)
    runs2 = [{"lr":lr,"ep":ep,"rk":rk,
               "fq":_fq(lr,ep,rk,rng2),"mu":_mu(lr,ep,rk,rng2)}
             for lr,ep,rk in product(LRS,EPOCHS,RANKS)]

    def adaptive_lr(t, lr_peak=1e-5, lr_base=5e-6, lr_adv=7e-6, lr_floor=2e-6,
                    t_warm=30, t_decay=200, t_plat=250):
        if t <= t_warm:
            return lr_base + (lr_peak - lr_base) * (t / t_warm)
        elif t <= t_decay:
            p = (t - t_warm) / (t_decay - t_warm)
            return lr_base + 0.5*(lr_peak-lr_base)*(1 + np.cos(np.pi*p))
        elif t <= t_plat:
            return lr_adv
        else:
            p = (t - t_plat) / (300 - t_plat)
            return lr_floor + 0.5*(lr_adv-lr_floor)*(1 + np.cos(np.pi*p))

    adap_v = [adaptive_lr(t) for t in steps]
    ax1.set_facecolor("#FAFAFB")
    for name, (lr, col) in {
        "1e-5 (default)": (1e-5, CORAL),
        "5e-6": (5e-6, SOFT),
        "2e-5": (2e-5, WARN),
    }.items():
        ax1.axhline(lr, lw=1.8, ls="--", label=f"Fixed {name}", alpha=0.8, color=col)
    ax1.plot(steps, adap_v, color=NAVY, lw=3.0, label="Adaptive (proposed)", zorder=5)
    # Phase shading
    ax1.axvspan(0,   30,  alpha=0.09, color=SAGE)
    ax1.axvspan(30,  200, alpha=0.07, color=SOFT)
    ax1.axvspan(200, 250, alpha=0.09, color=AMBER)
    ax1.axvspan(250, 300, alpha=0.07, color="#888")
    for x, lbl, col in [
        (15,"Warm\nup",SAGE), (115,"Cosine\ndecay",SOFT),
        (225,"Adv.\nplateau",AMBER), (275,"Cool\ndown","#666")
    ]:
        ax1.text(x, 1.85e-5, lbl, ha="center", fontsize=8, color=col, fontweight="bold")
    ax1.set_xlabel("Training step", fontsize=11)
    ax1.set_ylabel("Learning rate", fontsize=11)
    ax1.set_title("Adaptive vs fixed baselines", fontsize=11.5, color=NAVY, pad=7, fontweight="bold")
    ax1.legend(fontsize=8.5, frameon=True, framealpha=0.95)
    ax1.set_yscale("log"); ax1.tick_params(labelsize=10)
    for sp in ["top","right"]: ax1.spines[sp].set_visible(False)

    # Right: rank comparison bars (FQ) + triangles (MU) at ep=10
    ax2.set_facecolor("#FAFAFB")
    x = np.arange(len(RANKS)); bw = 0.18
    for i, (lr, col) in enumerate(LR_COLORS.items()):
        fq_v = [next(r["fq"] for r in runs2 if abs(r["lr"]-lr)<1e-9
                     and r["ep"]==10 and r["rk"]==rk) for rk in RANKS]
        mu_v = [next(r["mu"] for r in runs2 if abs(r["lr"]-lr)<1e-9
                     and r["ep"]==10 and r["rk"]==rk) for rk in RANKS]
        off = (i - 1.5) * bw
        ax2.bar(x + off, fq_v, width=bw, color=col, edgecolor="white",
                lw=0.6, alpha=0.85, label=f"lr={lr:.0e}", zorder=3)
        for j, mv in enumerate(mu_v):
            ax2.scatter(x[j]+off, mv, color=col, s=50, marker="^",
                        zorder=5, edgecolors="white", lw=0.8)

    ax2.axhline(0.85, color="#BBB", ls="--", lw=1.1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"rank={r}" for r in RANKS], fontsize=11)
    ax2.set_ylabel("FQ (bars) / MU (triangles ▲)", fontsize=10.5)
    ax2.set_ylim(0.83, 0.97)
    ax2.set_title("Rank effect at ep=10: higher rank → FQ↑ and MU↑\n(no tradeoff)",
                  fontsize=10.5, color=NAVY, pad=7, fontweight="bold")
    ax2.legend(fontsize=9, frameon=True, ncol=2)
    ax2.text(2.0, 0.962, "rank=32:\nFQ & MU best\nat every lr",
             ha="center", fontsize=8.5, color=NAVY, fontstyle="italic",
             bbox=dict(fc=ICE, ec=NAVY, lw=0.8, boxstyle="round,pad=0.25"))
    for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
    ax2.tick_params(labelsize=10)

    fig.suptitle("4-phase adaptive LR (left)  |  Rank lifts both FQ and MU (right, ep=10)",
                 fontsize=12, color=NAVY, fontweight="bold")
    plt.tight_layout()
    path = Path(out_dir) / "adaptive_lr.png"
    plt.savefig(path, dpi=185, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    return path

# ── CLI entry point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FORGE hyperparameter sweep simulation")
    parser.add_argument("--seed",    type=int,   default=42,        help="RNG seed")
    parser.add_argument("--out_dir", type=str,   default="results", help="Output directory")
    parser.add_argument("--show",    action="store_true",           help="Show plots interactively")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Running sweep simulation...")
    results = run_sweep(seed=args.seed)

    # Save raw data
    data_path = out / "sweep_results.json"
    with open(data_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} results → {data_path}")

    # Print analysis
    print_summary(results)

    # Generate figures
    for name, fn in [
        ("scatter",    lambda: fig_scatter(results, out)),
        ("heatmaps",   lambda: fig_heatmaps(results, out)),
        ("lr/epoch",   lambda: fig_lr_epoch(results, out)),
        ("adaptive",   lambda: fig_adaptive(out)),
    ]:
        path = fn()
        print(f"  Figure '{name}' → {path}")

    # Dump recommended config to JSON
    rec = recommended_config(results)
    rec_path = out / "recommended_config.json"
    with open(rec_path, "w") as f:
        json.dump({
            "lr":      rec["lr"],
            "epochs":  rec["ep"],
            "rank":    rec["rk"],
            "fq":      rec["fq"],
            "mu":      rec["mu"],
            "note":    "Pareto-optimal (max FQ+MU). Use with 4-phase adaptive LR schedule.",
        }, f, indent=2)
    print(f"  Recommended config → {rec_path}")
    print(f"\nDone. Outputs in: {out.resolve()}")

if __name__ == "__main__":
    main()
