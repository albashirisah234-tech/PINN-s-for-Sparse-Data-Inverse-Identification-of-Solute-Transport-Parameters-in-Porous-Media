"""
experiments/run_level3_sparsity_sweep.py
==========================================
Level 3: the full sparsity sweep. For every combination of experiment (10),
sparsity level (6), and random seed (3 -- uniform across all levels, see
note below), trains both an inverse PINN and a data-only NN baseline, then
scores:
  - outlet-curve reconstruction (RMSE/R2) for both models
  - parameter-recovery error (v, D) for the PINN only
  - spatial generalization at 3 unobserved interior locations (PINN only --
    the data-only baseline has no spatial input and cannot attempt this)
  - final training loss, used afterward for the best-of-K analysis

WHY 3 SEEDS AT EVERY LEVEL (not just the sparse ones):
An earlier version of this sweep spent more seeds on sparse levels only,
assuming dense levels were seed-stable. A smoke test disproved that:
dispersion-recovery error was clearly BIMODAL at every sparsity level
tested, including full resolution -- every run landed either ~1-2% error
or ~15-24% error, regardless of how much data it saw. Seeds are therefore
uniform across all levels so the sweep can characterize this properly
rather than assume it away.

Runtime: ~68s per (experiment, sparsity, seed) combination on a single GPU.
Full sweep = 10 experiments x 6 levels x 3 seeds = 180 runs ~ 3.4 hours.
Use --smoke-test first (12-18 runs, ~15-20 min) to sanity-check before
committing to the full run.

Usage:
    python experiments/run_level3_sparsity_sweep.py --data data/Homogeneous_MoM_BTC.xlsx --smoke-test
    python experiments/run_level3_sparsity_sweep.py --data data/Homogeneous_MoM_BTC.xlsx --epochs 5000

IMPORTANT: download/copy the output files from results/ immediately after
this finishes, especially on ephemeral environments (e.g. Kaggle's
/kaggle/working/ does not reliably persist between sessions) -- do not
navigate away or start another task first.
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_io import load_curve, subsample_fixed_count, EXPERIMENTS, KIND_OF
from src.analytical import fit_ground_truth_fixedL, analytical_at_x, L_KNOWN
from src.training import train_inverse_pinn, train_data_only_nn, device

SPARSITY_LEVELS = [None, 500, 100, 50, 20, 10]
INTERIOR_X_TEST = [0.25, 0.50, 0.75]  # nondim locations, never measured in real data


def run_sweep(data_path, epochs, n_seeds, smoke_test, out_dir):
    exp_list = ["Exp 7"] if smoke_test else EXPERIMENTS
    results = []

    for exp in exp_list:
        kind = KIND_OF[exp]
        t_raw, c_raw = load_curve(exp, data_path)
        T_known = t_raw.max()
        t_full_nondim = t_raw / T_known

        v_gt, D_gt = fit_ground_truth_fixedL(t_raw, c_raw, kind)
        print(f"\n=== {exp} ({kind}) === ground truth (L={L_KNOWN}m fixed): "
              f"v={v_gt:.4e} m/s, D={D_gt:.4e} m^2/s")

        for n_pts in SPARSITY_LEVELS:
            t_sub, c_sub = subsample_fixed_count(t_raw, c_raw, n_pts)
            t_sub_nondim = t_sub / T_known
            n_actual = len(t_sub)
            label = "full" if n_pts is None else str(n_pts)

            for seed in range(n_seeds):
                t0 = time.time()

                pinn, final_loss = train_inverse_pinn(t_sub_nondim, c_sub, kind,
                                                       seed=seed, n_epochs=epochs)
                v_learned = pinn.v.item() * L_KNOWN / T_known
                D_learned = pinn.D.item() * L_KNOWN ** 2 / T_known

                with torch.no_grad():
                    x_eval = torch.ones(len(t_full_nondim), 1).to(device)
                    t_eval = torch.tensor(t_full_nondim, dtype=torch.float32).unsqueeze(1).to(device)
                    c_pred_pinn = pinn(x_eval, t_eval).cpu().numpy().flatten()
                rmse_pinn = np.sqrt(np.mean((c_pred_pinn - c_raw) ** 2))
                r2_pinn = 1 - np.sum((c_pred_pinn - c_raw) ** 2) / np.sum((c_raw - c_raw.mean()) ** 2)

                # spatial generalization: PINN at unobserved interior x
                interior_rmses = []
                with torch.no_grad():
                    for x_test in INTERIOR_X_TEST:
                        x_phys = x_test * L_KNOWN
                        c_analytical_interior = analytical_at_x(x_phys, t_raw, v_gt, D_gt, kind)
                        x_in = torch.full((len(t_full_nondim), 1), x_test).to(device)
                        c_pinn_interior = pinn(x_in, t_eval).cpu().numpy().flatten()
                        interior_rmses.append(np.sqrt(np.mean((c_pinn_interior - c_analytical_interior) ** 2)))
                interior_rmse_mean = float(np.mean(interior_rmses))

                # data-only NN baseline (outlet-only, cannot do interior test)
                dnn = train_data_only_nn(t_sub_nondim, c_sub, seed=seed, n_epochs=epochs)
                with torch.no_grad():
                    c_pred_dnn = dnn(t_eval).cpu().numpy().flatten()
                rmse_dnn = np.sqrt(np.mean((c_pred_dnn - c_raw) ** 2))
                r2_dnn = 1 - np.sum((c_pred_dnn - c_raw) ** 2) / np.sum((c_raw - c_raw.mean()) ** 2)

                elapsed = time.time() - t0
                print(f"  n={label:>5s} seed={seed}  "
                      f"PINN: RMSE={rmse_pinn:.4f} v_err={abs(v_learned-v_gt)/v_gt*100:5.1f}% "
                      f"D_err={abs(D_learned-D_gt)/D_gt*100:5.1f}% interior_RMSE={interior_rmse_mean:.4f} "
                      f"final_loss={final_loss:.3e}  |  DataNN: RMSE={rmse_dnn:.4f}   [{elapsed:.1f}s]")

                results.append(dict(
                    exp=exp, kind=kind, n_points_requested=label, n_points_actual=n_actual, seed=seed,
                    v_gt=v_gt, D_gt=D_gt, v_pinn=v_learned, D_pinn=D_learned,
                    v_err_pct=abs(v_learned - v_gt) / v_gt * 100, D_err_pct=abs(D_learned - D_gt) / D_gt * 100,
                    rmse_pinn=rmse_pinn, r2_pinn=r2_pinn, final_loss=final_loss,
                    interior_rmse_pinn=interior_rmse_mean,
                    rmse_dnn=rmse_dnn, r2_dnn=r2_dnn,
                ))

    df = pd.DataFrame(results)
    raw_csv = os.path.join(out_dir, "tables", "level3_sparsity_results_raw.csv")
    df.to_csv(raw_csv, index=False)

    agg = df.groupby(["exp", "kind", "n_points_requested"], observed=True).agg(
        n_points_actual=("n_points_actual", "first"),
        v_err_mean=("v_err_pct", "mean"), v_err_std=("v_err_pct", "std"),
        D_err_mean=("D_err_pct", "mean"), D_err_std=("D_err_pct", "std"),
        rmse_pinn_mean=("rmse_pinn", "mean"), rmse_pinn_std=("rmse_pinn", "std"),
        interior_rmse_mean=("interior_rmse_pinn", "mean"), interior_rmse_std=("interior_rmse_pinn", "std"),
        rmse_dnn_mean=("rmse_dnn", "mean"), rmse_dnn_std=("rmse_dnn", "std"),
    ).reset_index()

    # best-of-K (lowest final loss): the practical, ground-truth-free
    # selection protocol -- Section 3.5 of the paper tests whether this is
    # actually reliable (it is not: ~48% success rate across 60 combos)
    best_of_k = df.loc[df.groupby(["exp", "n_points_requested"], observed=True)["final_loss"].idxmin()]
    best_of_k = best_of_k[["exp", "n_points_requested", "seed", "final_loss", "v_err_pct", "D_err_pct"]]
    best_of_k = best_of_k.rename(columns={"seed": "best_seed", "v_err_pct": "best_of_k_v_err",
                                           "D_err_pct": "best_of_k_D_err"})
    agg = agg.merge(best_of_k, on=["exp", "n_points_requested"])
    agg_csv = os.path.join(out_dir, "tables", "level3_sparsity_results_aggregated.csv")
    agg.to_csv(agg_csv, index=False)

    print(f"\nSaved {len(df)} raw rows -> {raw_csv}")
    print(f"Saved {len(agg)} aggregated rows -> {agg_csv}")
    print("\nBest-of-K (lowest training loss) vs mean-across-seeds D error:")
    for _, row in agg.iterrows():
        print(f"  {row['exp']:8s} n={row['n_points_requested']:>5s}  "
              f"best-of-K D_err={row['best_of_k_D_err']:5.1f}% (seed {row['best_seed']})  "
              f"vs mean D_err={row['D_err_mean']:5.1f}%")
    return df, agg


def plot_summary(agg, out_png):
    order = ["10", "20", "50", "100", "500", "full"]
    agg["n_points_requested"] = pd.Categorical(agg["n_points_requested"], categories=order, ordered=True)
    summary = agg.groupby("n_points_requested", observed=True).mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x_labels = summary["n_points_requested"].astype(str)

    axes[0].errorbar(x_labels, summary["rmse_pinn_mean"], yerr=summary["rmse_pinn_std"], marker="o", label="PINN (outlet)")
    axes[0].plot(x_labels, summary["rmse_dnn_mean"], "s--", label="Data-only NN (outlet)")
    axes[0].set_title("Outlet reconstruction RMSE"); axes[0].set_xlabel("training points"); axes[0].legend()

    axes[1].errorbar(x_labels, summary["v_err_mean"], yerr=summary["v_err_std"], marker="o", label="v error (%)")
    axes[1].errorbar(x_labels, summary["D_err_mean"], yerr=summary["D_err_std"], marker="s", label="D error (%)")
    axes[1].set_title("Parameter recovery error (mean +/- std)"); axes[1].set_xlabel("training points"); axes[1].legend()

    axes[2].errorbar(x_labels, summary["interior_rmse_mean"], yerr=summary["interior_rmse_std"], marker="o",
                      color="C2", label="PINN interior-x RMSE\n(data-only NN cannot do this)")
    axes[2].set_title("Spatial generalization\nvs analytical solution"); axes[2].set_xlabel("training points"); axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Homogeneous_MoM_BTC.xlsx")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true",
                         help="Restrict to Exp 7 only (all sparsity levels/seeds) for a quick check")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    print(f"Using device: {device}")

    df, agg = run_sweep(args.data, args.epochs, args.seeds, args.smoke_test, args.out_dir)
    plot_summary(agg, os.path.join(args.out_dir, "figures", "level3_sparsity_summary.png"))
