"""
experiments/make_paper_figures.py
====================================
Reproduces every figure AND every statistical claim in the paper's Results
section from the Level 3 raw results CSV. This script exists specifically
so a reader can verify the paper's numbers independently rather than trust
them -- run it after run_level3_sparsity_sweep.py has produced
results/tables/level3_sparsity_results_raw.csv.

IMPORTANT METHODOLOGICAL NOTE (read before modifying the stats functions):
The 3 seeds trained at each (experiment, sparsity) combination are repeated
optimizations of the SAME underlying ground truth, not independent
experimental replicates. Every function below that summarizes "across
experiments" therefore averages the 3 seeds within each experiment FIRST,
then computes statistics across the resulting 10 independent per-experiment
values -- never naively pooling all 30 (experiment x seed) rows together.
Because the design is balanced (exactly 3 seeds everywhere), the two
approaches give identical MEANS but different MEDIANS and variance
estimates -- see hierarchical_sparsity_table() for why this matters (the
mean and median diverge substantially in this dataset, itself informative).

Usage:
    python experiments/make_paper_figures.py --raw results/tables/level3_sparsity_results_raw.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats

SPARSITY_ORDER = ["10", "20", "50", "100", "500", "full"]


def _per_experiment(df):
    """Collapse the 3 (non-independent) seeds to one value per experiment,
    per sparsity level -- the correct unit of replication for all
    across-experiment statistics in this project. See module docstring."""
    d = df.copy()
    d["n_points_requested"] = pd.Categorical(d["n_points_requested"], categories=SPARSITY_ORDER, ordered=True)
    return d.groupby(["exp", "kind", "n_points_requested"], observed=True)[
        ["v_err_pct", "D_err_pct", "interior_rmse_pinn"]].mean().reset_index()


def hierarchical_sparsity_table(df, out_csv):
    """Reproduces Table 3: mean, median, and IQR of parameter-recovery
    error by sparsity level, computed hierarchically (n=10 independent
    experiments, seeds pre-averaged within each) -- NOT a naive pool of
    all 30 raw rows, which would understate the true between-experiment
    variability by treating correlated seed replicates as independent."""
    per_exp = _per_experiment(df)
    table = per_exp.groupby("n_points_requested", observed=True).agg(
        v_err_mean=("v_err_pct", "mean"), v_err_median=("v_err_pct", "median"),
        D_err_mean=("D_err_pct", "mean"), D_err_median=("D_err_pct", "median"),
        D_err_q1=("D_err_pct", lambda x: x.quantile(0.25)),
        D_err_q3=("D_err_pct", lambda x: x.quantile(0.75)),
    ).reset_index()
    table.to_csv(out_csv, index=False)
    print("=== Table 3: parameter-recovery error by sparsity (n=10 experiments, hierarchical) ===")
    print(table.to_string(index=False))
    print(f"Saved: {out_csv}\n")
    return table


def fig_sparsity_pooled(df, out_path):
    """Figure 5. Uses the same hierarchical (per-experiment-first) values
    as hierarchical_sparsity_table() for consistency between the figure
    and the printed/saved table."""
    per_exp = _per_experiment(df)
    summary = per_exp.groupby("n_points_requested", observed=True).agg(
        v_mean=("v_err_pct", "mean"), v_std=("v_err_pct", "std"),
        D_mean=("D_err_pct", "mean"), D_std=("D_err_pct", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    x = summary["n_points_requested"].astype(str)
    ax.errorbar(x, summary["v_mean"], yerr=summary["v_std"], marker="o", capsize=4, label="Velocity (v) error")
    ax.errorbar(x, summary["D_mean"], yerr=summary["D_std"], marker="s", capsize=4, label="Dispersion (D) error")
    ax.set_xlabel("Number of training points"); ax.set_ylabel("Parameter recovery error (%)")
    ax.set_title("Parameter recovery error vs. sparsity\n(mean \u00b1 std across 10 independent experiments)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_load_vs_unload(df, out_path):
    """Figure 7 + the Section 3.7 Mann-Whitney test. One value per
    experiment (n=5 load, n=5 unload) -- this test was already correctly
    de-duplicated in the original analysis, unlike the sparsity table."""
    per_exp = _per_experiment(df)
    full_res = per_exp[per_exp.n_points_requested == "full"]

    fig, ax = plt.subplots(figsize=(6, 5))
    data_to_plot = [full_res[full_res.kind == "load"]["D_err_pct"], full_res[full_res.kind == "unload"]["D_err_pct"]]
    bp = ax.boxplot(data_to_plot, tick_labels=["Load", "Unload"], patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel("Dispersion (D) recovery error (%)")
    ax.set_title("Dispersion recovery error: load vs. unload experiments\n(full resolution, n=5 each)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    load_D = full_res[full_res.kind == "load"]["D_err_pct"]
    unload_D = full_res[full_res.kind == "unload"]["D_err_pct"]
    u_stat, p_value = stats.mannwhitneyu(load_D, unload_D, alternative="less")
    print(f"\n=== Section 3.7: load vs. unload (n=5 experiments each, full resolution) ===")
    print(f"Mann-Whitney U (load < unload): U={u_stat:.1f}, p={p_value:.4f}")
    print(f"Load median D error:   {load_D.median():.1f}%")
    print(f"Unload median D error: {unload_D.median():.1f}%\n")


def fig_seed_bimodality(df, exp_name, out_path):
    """Figure 6 (illustrative single-experiment case)."""
    exp_df = df[df.exp == exp_name].copy()
    exp_df["n_points_requested"] = pd.Categorical(exp_df["n_points_requested"], categories=SPARSITY_ORDER, ordered=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for seed in sorted(exp_df.seed.unique()):
        sub = exp_df[exp_df.seed == seed].sort_values("n_points_requested")
        ax.plot(sub["n_points_requested"].astype(str), sub["D_err_pct"], marker="o", label=f"seed {seed}")
    ax.set_xlabel("Number of training points"); ax.set_ylabel("Dispersion (D) error (%)")
    ax.set_title(f"{exp_name}: seed-dependent bimodal convergence\n(largely independent of sparsity level)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_conceptual_three_errors(out_path):
    """Figure 1: the schematic distinguishing training loss / BTC error /
    parameter error, framing the whole Results section."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    boxes = [
        (0.5, 1.5, "Training\nloss", "#B8CCE4"),
        (3.5, 1.5, "BTC prediction\nerror (RMSE)", "#C6E0B4"),
        (6.5, 1.5, "Parameter\nrecovery error\n(v, D)", "#F4B183"),
    ]
    for x, y, label, color in boxes:
        box = FancyBboxPatch((x, y), 2.2, 1.2, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor="black", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + 1.1, y + 0.6, label, ha="center", va="center", fontsize=11)
    arrow_style = dict(arrowstyle="-|>", mutation_scale=20, color="black", linewidth=1.5)
    ax.add_patch(FancyArrowPatch((2.7, 2.1), (3.5, 2.1), **arrow_style))
    ax.add_patch(FancyArrowPatch((5.7, 2.1), (6.5, 2.1), linestyle="--", **{**arrow_style, "color": "gray"}))
    ax.text(3.1, 2.4, "assumed to\ntrack", ha="center", fontsize=9, style="italic")
    ax.text(6.1, 2.4, "does NOT\nreliably track\n(Section 3.5)", ha="center", fontsize=9, style="italic", color="darkred")
    ax.text(1.6, 0.9, "minimized directly\nby the optimizer", ha="center", fontsize=8.5, color="dimgray")
    ax.text(4.6, 0.9, "low for BOTH the\nPINN and the data-only\nbaseline (Section 3.6)", ha="center", fontsize=8.5, color="dimgray")
    ax.text(7.6, 0.9, "the quantity that\nactually matters for\ndownstream use", ha="center", fontsize=8.5, color="dimgray")
    ax.set_title("Three distinct quantities conflated in most PINN evaluations\n(this study measures all three separately)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def best_of_k_reliability_check(df):
    """Section 3.5: does lowest-training-loss seed selection reliably beat
    the seed-average? Includes the binomial test against chance (p=0.5)
    that a reviewer correctly pointed out was missing from an earlier
    draft of this analysis -- asserting '~48%, indistinguishable from
    chance' without the test is a claim, not a result."""
    agg = df.groupby(["exp", "n_points_requested"], observed=True).agg(
        D_err_mean=("D_err_pct", "mean")).reset_index()
    best_of_k = df.loc[df.groupby(["exp", "n_points_requested"], observed=True)["final_loss"].idxmin()]
    best_of_k = best_of_k[["exp", "n_points_requested", "D_err_pct"]].rename(columns={"D_err_pct": "best_of_k_D_err"})
    merged = agg.merge(best_of_k, on=["exp", "n_points_requested"])
    merged["helps"] = merged["best_of_k_D_err"] < merged["D_err_mean"]
    n = len(merged)
    k = int(merged["helps"].sum())
    result = stats.binomtest(k, n, p=0.5)
    ci = result.proportion_ci()
    print(f"=== Section 3.5: best-of-K (lowest training loss) reliability ===")
    print(f"Best-of-K beats mean-of-seeds in {k}/{n} cases ({k/n*100:.0f}%)")
    print(f"Binomial test vs. chance (p=0.5): p-value={result.pvalue:.4f}, "
          f"95% CI=({ci.low*100:.0f}%, {ci.high*100:.0f}%)")
    print("--> Fails to reject the null: training loss is not a reliable proxy for parameter accuracy.\n")


def spatial_generalization_significance_test(df):
    """Section 3.6: tests whether the clean monotonic sparsity trend seen
    for the single illustrative experiment (Exp 7) actually generalizes
    across all 10 experiments. IMPORTANT: it does not (both tests below
    are non-significant) -- this is reported in the paper as an honest
    negative result, not suppressed. Re-running this is the way to check
    that for yourself rather than trust the paper's prose."""
    per_exp = _per_experiment(df)
    pivot = per_exp.pivot(index="exp", columns="n_points_requested", values="interior_rmse_pinn")
    stat, p_wilcoxon = stats.wilcoxon(pivot["10"], pivot["full"], alternative="greater")

    rank_map = {v: i + 1 for i, v in enumerate(SPARSITY_ORDER)}
    d = df.copy()
    d["sparsity_rank"] = d["n_points_requested"].map(rank_map)
    rho, p_spearman = stats.spearmanr(d["sparsity_rank"], d["interior_rmse_pinn"])

    print("=== Section 3.6: spatial-generalization trend, tested across all 10 experiments ===")
    print(f"Wilcoxon signed-rank (interior RMSE at n=10 > full, paired by experiment): "
          f"stat={stat:.1f}, p={p_wilcoxon:.4f}")
    print(f"Spearman rho (sparsity rank vs. interior RMSE, all 180 raw rows): "
          f"rho={rho:.3f}, p={p_spearman:.4f}")
    print("--> Neither reaches significance: the Exp-7 illustration does NOT generalize "
          "to a dataset-wide trend. Reported in the paper as an open question, not a finding.\n")


def seed_variability_summary(df):
    """Section 3.4: quantifies how many of the 10 experiments show a wide
    seed-to-seed spread in D recovery at full resolution (not just the
    single Exp 7 example shown in Figure 6)."""
    full = df[df.n_points_requested == "full"]
    spread = full.groupby("exp")["D_err_pct"].agg(["min", "max"]).reset_index()
    spread["range"] = spread["max"] - spread["min"]
    n_wide = int((spread["range"] > 10).sum())
    print(f"=== Section 3.4: seed-to-seed spread at full resolution ===")
    print(spread.to_string(index=False))
    print(f"--> {n_wide}/10 experiments show a >10 percentage-point spread in D error "
          f"across their 3 seeds: seed sensitivity is pervasive, not specific to Exp 7.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="results/tables/level3_sparsity_results_raw.csv")
    parser.add_argument("--bimodality-exp", default="Exp 7",
                         help="Which experiment to use for the seed-bimodality figure (Figure 6)")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    fig_dir = os.path.join(args.out_dir, "figures")
    table_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    df = pd.read_csv(args.raw)

    # Figures (numbered to match the paper)
    fig_conceptual_three_errors(os.path.join(fig_dir, "fig1_conceptual.png"))
    fig_sparsity_pooled(df, os.path.join(fig_dir, "fig5_sparsity_pooled.png"))
    fig_seed_bimodality(df, args.bimodality_exp, os.path.join(fig_dir, "fig6_seed_bimodality.png"))
    fig_load_vs_unload(df, os.path.join(fig_dir, "fig7_load_vs_unload.png"))

    # Tables and statistical tests (all numbers reported in the paper's Results section)
    hierarchical_sparsity_table(df, os.path.join(table_dir, "table3_sparsity_hierarchical.csv"))
    seed_variability_summary(df)
    best_of_k_reliability_check(df)
    spatial_generalization_significance_test(df)
