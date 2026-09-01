"""
experiments/run_level1_validation.py
======================================
Level 1: build the master ground-truth parameter table for all 10
homogeneous experiments (analytical fit, L=0.03m fixed), and validate the
Crank-Nicolson finite-difference solver against both the analytical
solution and the raw experimental data for one representative experiment.

Usage:
    python experiments/run_level1_validation.py --data data/Homogeneous_MoM_BTC.xlsx
    python experiments/run_level1_validation.py --data data/Homogeneous_MoM_BTC.xlsx --exp "Exp 3"

Outputs (written to results/):
    results/tables/ground_truth_parameters.csv
    results/figures/level1_validation.png
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_io import load_curve, EXPERIMENTS, KIND_OF
from src.analytical import fit_ground_truth_fixedL, ogata_banks_fixedL, L_KNOWN
from src.numerical import solve_ade_numerical


def build_ground_truth_table(data_path, out_csv):
    rows = []
    for exp in EXPERIMENTS:
        t, c = load_curve(exp, data_path)
        kind = KIND_OF[exp]
        v, D = fit_ground_truth_fixedL(t, c, kind)
        model = ogata_banks_fixedL if kind == 'load' else (lambda t, v, D: 1.0 - ogata_banks_fixedL(t, v, D))
        pred = model(t, v, D)
        r2 = 1 - np.sum((pred - c) ** 2) / np.sum((c - c.mean()) ** 2)
        rmse = np.sqrt(np.mean((pred - c) ** 2))
        rows.append(dict(exp=exp, kind=kind, n_points=len(t), v=v, D=D, R2=r2, RMSE=rmse))
        print(f"{exp:8s} ({kind:6s}) n={len(t):5d}  v={v:.4e}  D={D:.4e}  R2={r2:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved ground-truth table: {out_csv}")
    return df


def validate_one_experiment(data_path, exp_name, out_png):
    t, c = load_curve(exp_name, data_path)
    kind = KIND_OF[exp_name]
    v, D = fit_ground_truth_fixedL(t, c, kind)

    model = ogata_banks_fixedL if kind == 'load' else (lambda t, v, D: 1.0 - ogata_banks_fixedL(t, v, D))
    pred_analytical = model(t, v, D)
    r2 = 1 - np.sum((pred_analytical - c) ** 2) / np.sum((c - c.mean()) ** 2)

    t_num, c_num = solve_ade_numerical(v, D, L_KNOWN, t[t > 0])
    c_analytical_at_num_t = model(t_num, v, D)
    rmse_num_vs_analytical = np.sqrt(np.mean((c_num - c_analytical_at_num_t) ** 2))
    interp_data = np.interp(t_num, t, c)
    rmse_num_vs_data = np.sqrt(np.mean((c_num - interp_data) ** 2))

    print(f"\n{exp_name}: analytical fit R2={r2:.4f}")
    print(f"Numerical vs analytical RMSE: {rmse_num_vs_analytical:.5f}")
    print(f"Numerical vs experimental data RMSE: {rmse_num_vs_data:.5f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(t, c, s=8, alpha=0.4, color="gray", label=f"Experimental BTC ({exp_name})")
    ax.plot(t, pred_analytical, label=f"Analytical (Ogata-Banks) fit, R\u00b2={r2:.4f}", color="C0", lw=2)
    ax.plot(t_num, c_num, "--", label="Numerical (Crank-Nicolson FD)", color="C3", lw=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("C / C0")
    ax.set_title("Level 1 validation: numerical vs analytical vs experimental BTC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Homogeneous_MoM_BTC.xlsx",
                         help="Path to the homogeneous-set BTC spreadsheet")
    parser.add_argument("--exp", default="Exp 7",
                         help="Which experiment to use for the FD validation plot")
    parser.add_argument("--out-dir", default="results", help="Output root directory")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)

    build_ground_truth_table(args.data, os.path.join(args.out_dir, "tables", "ground_truth_parameters.csv"))
    validate_one_experiment(args.data, args.exp, os.path.join(args.out_dir, "figures", "level1_validation.png"))
