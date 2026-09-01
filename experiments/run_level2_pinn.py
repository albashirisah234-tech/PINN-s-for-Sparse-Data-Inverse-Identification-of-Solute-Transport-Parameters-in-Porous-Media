"""
experiments/run_level2_pinn.py
================================
Level 2a (forward) and Level 2b (inverse), single representative
experiment. Confirms the PINN training pipeline works correctly before
committing to the full Level 3 sweep.

Usage:
    python experiments/run_level2_pinn.py --data data/Homogeneous_MoM_BTC.xlsx --mode forward --epochs 8000
    python experiments/run_level2_pinn.py --data data/Homogeneous_MoM_BTC.xlsx --mode inverse --epochs 10000

Outputs (written to results/):
    results/figures/level2_forward_validation.png   (--mode forward)
    results/figures/level2_inverse_full_res.png      (--mode inverse)
"""
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import erfc, erfcx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_io import load_curve, KIND_OF
from src.analytical import fit_ground_truth_fixedL, L_KNOWN
from src.training import train_forward_pinn, train_inverse_pinn, device


def analytical_nondim(xp, tp, Pe):
    """Ogata-Banks in nondimensional (x', t') with v'=1, D'=1/Pe -- used
    only for validating the FORWARD PINN, which is allowed to know v, D."""
    tp = np.clip(tp, 1e-10, None)
    Dp = 1.0 / Pe
    sqrt_Dt = np.sqrt(Dp * tp)
    z1 = (xp - tp) / (2 * sqrt_Dt)
    z2 = (xp + tp) / (2 * sqrt_Dt)
    return 0.5 * (erfc(z1) + erfcx(z2) * np.exp(-z1**2))


def run_forward(data_path, exp_name, epochs, out_png):
    t, c = load_curve(exp_name, data_path)
    kind = KIND_OF[exp_name]
    v_true, D_true = fit_ground_truth_fixedL(t, c, kind)
    L = L_KNOWN
    T = L / v_true
    Pe_true = v_true * L / D_true
    print(f"Ground truth: v={v_true:.4e} m/s, D={D_true:.4e} m^2/s, Pe={Pe_true:.2f}")

    t_max_dimless = 4.0
    inv_Pe_true = 1.0 / Pe_true
    model = train_forward_pinn(v_true_dimless=1.0, D_true_dimless=inv_Pe_true,
                                n_epochs=epochs, t_max_dimless=t_max_dimless)

    xg = np.linspace(0, 1, 100)
    tg = np.linspace(0.01, t_max_dimless, 100)
    Xg, Tg = np.meshgrid(xg, tg)
    with torch.no_grad():
        xt = torch.tensor(Xg.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
        tt = torch.tensor(Tg.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
        C_pred = model(xt, tt).cpu().numpy().reshape(Xg.shape)
    C_analytical = analytical_nondim(Xg, Tg, Pe_true)

    rmse = np.sqrt(np.mean((C_pred - C_analytical) ** 2))
    r2 = 1 - np.sum((C_pred - C_analytical) ** 2) / np.sum((C_analytical - C_analytical.mean()) ** 2)
    print(f"Full-field validation vs analytical: RMSE={rmse:.5f} R2={r2:.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(tg, C_analytical[:, -1], label="Analytical (outlet)", lw=2)
    axes[0].plot(tg, C_pred[:, -1], "--", label="PINN (outlet)", lw=2)
    axes[0].set_xlabel("t' (dimensionless)"); axes[0].set_ylabel("C/C0")
    axes[0].set_title(f"Forward PINN vs analytical, outlet BTC\n{exp_name}")
    axes[0].legend()
    im = axes[1].imshow(np.abs(C_pred - C_analytical), extent=[0, 1, t_max_dimless, 0.01],
                         aspect="auto", cmap="viridis")
    axes[1].set_xlabel("x' (dimensionless)"); axes[1].set_ylabel("t' (dimensionless)")
    axes[1].set_title(f"|PINN - analytical| full-field error\nRMSE={rmse:.4f}")
    plt.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


def run_inverse(data_path, exp_name, epochs, out_png):
    t_raw, c_raw = load_curve(exp_name, data_path)
    kind = KIND_OF[exp_name]
    T_known = t_raw.max()
    t_nondim = t_raw / T_known

    v_gt, D_gt = fit_ground_truth_fixedL(t_raw, c_raw, kind)
    print(f"[reference only, NOT used in training] v_gt={v_gt:.4e} D_gt={D_gt:.4e}")

    model, final_loss = train_inverse_pinn(t_nondim, c_raw, kind, n_epochs=epochs)
    v_learned = model.v.item() * L_KNOWN / T_known
    D_learned = model.D.item() * L_KNOWN ** 2 / T_known
    print(f"Recovered:  v={v_learned:.4e}  D={D_learned:.4e}")
    print(f"Rel. error: v={abs(v_learned-v_gt)/v_gt*100:.1f}%  D={abs(D_learned-D_gt)/D_gt*100:.1f}%")

    with torch.no_grad():
        t_plot = torch.linspace(0, 1, 300).unsqueeze(1).to(device)
        x_plot = torch.ones_like(t_plot)
        c_plot = model(x_plot, t_plot).cpu().numpy().flatten()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(t_raw, c_raw, s=8, alpha=0.4, color="gray", label="Experimental data (full res)")
    ax.plot(t_plot.cpu().numpy() * T_known, c_plot, color="C3", lw=2,
            label=f"Inverse PINN fit (v={v_learned:.2e}, D={D_learned:.2e})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("C/C0")
    ax.set_title(f"Inverse PINN, {exp_name}, full resolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Homogeneous_MoM_BTC.xlsx")
    parser.add_argument("--exp", default="Exp 7")
    parser.add_argument("--mode", choices=["forward", "inverse"], required=True)
    parser.add_argument("--epochs", type=int, default=None,
                         help="Default: 8000 for forward, 10000 for inverse")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    print(f"Using device: {device}")

    if args.mode == "forward":
        run_forward(args.data, args.exp, args.epochs or 8000,
                    os.path.join(args.out_dir, "figures", "level2_forward_validation.png"))
    else:
        run_inverse(args.data, args.exp, args.epochs or 10000,
                    os.path.join(args.out_dir, "figures", "level2_inverse_full_res.png"))
