"""
src/analytical.py
===================
The analytical Ogata-Banks solution to the 1D advection-dispersion equation
(Ogata & Banks, 1961, https://doi.org/10.3133/pp411A), and the curve-fitting
routines used to establish ground-truth (v, D) for each experiment.

IMPORTANT METHODOLOGICAL NOTE (read before changing L_KNOWN):
The deposited spreadsheet data give no units for x, t, v, D. Section 2.3 of
the accompanying paper recovers them empirically: fitting this analytical
solution with the sensor location x left as a FREE parameter converges to
0.025-0.036 m across all ten curves, matching the source paper's stated 3 cm
micromodel length -- confirming x in meters, t in seconds, v in m/s, D in
m^2/s.

For ground-truth values used to SCORE inverse-PINN recovery (as opposed to
this exploratory unit-recovery step), L is instead FIXED at 0.03 m
(L_KNOWN below) rather than left floating. This matches the fixed-L
convention the inverse PINN itself must use (it cannot leak the unknown x
into its own non-dimensionalization -- see src/models.py), so that recovery
error reflects genuine model performance rather than a mismatched length
assumption. Fitting with floating x (fit_floating_x) is provided for
completeness/validation but is NOT what ground-truth values in this
project are computed from.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc, erfcx

L_KNOWN = 0.03  # meters; paper-reported micromodel length (fixed convention)


def ogata_banks(t, v, D, x, C0=1.0):
    """Numerically stable Ogata-Banks solution for a step inlet input.

    Uses erfcx (scaled complementary error function) to avoid the overflow
    that a naive exp(x*v/D)*erfc(...) formulation produces for realistic
    parameter magnitudes -- this is not optional, the naive form silently
    fails (returns inf/nan, or worse, appears to converge to the initial
    guess with curve_fit reporting no error).
    """
    t = np.clip(t, 1e-10, None)
    sqrt_Dt = np.sqrt(D * t)
    z1 = (x - v * t) / (2 * sqrt_Dt)
    z2 = (x + v * t) / (2 * sqrt_Dt)
    return 0.5 * C0 * (erfc(z1) + erfcx(z2) * np.exp(-z1**2))


def ogata_banks_fixedL(t, v, D, x=L_KNOWN, C0=1.0):
    """Same as ogata_banks but with x defaulted to L_KNOWN, for use directly
    as a curve_fit model function (curve_fit needs the varying args first)."""
    return ogata_banks(t, v, D, x, C0)


def analytical_at_x(x_phys, t_phys, v, D, kind):
    """Evaluate the analytical solution at an arbitrary physical location
    x_phys, for either 'load' or 'unload' kind (unload = C0 - load, by
    linearity of the governing PDE -- see paper Section 2.2)."""
    c = ogata_banks(t_phys, v, D, x=x_phys)
    return c if kind == 'load' else (1.0 - c)


def fit_ground_truth_fixedL(t, c, kind):
    """Fit (v, D) with L fixed at L_KNOWN -- the ground-truth convention
    used throughout this project to score inverse-PINN recovery."""
    model = ogata_banks_fixedL if kind == 'load' else \
        (lambda t, v, D: 1.0 - ogata_banks_fixedL(t, v, D))
    popt, _ = curve_fit(model, t, c, p0=[1e-3, 1e-6],
                         bounds=([1e-7, 1e-10], [1e-1, 1e-3]), maxfev=50000)
    return popt  # (v, D)


def fit_floating_x(t, c, kind, p0=(5e-4, 5e-7, 0.025)):
    """Fit (v, D, x) with x free -- used only to recover measurement units
    (Section 2.3), NOT for ground-truth scoring. Returns (v, D, x, R2)."""
    model = ogata_banks if kind == 'load' else \
        (lambda t, v, D, x: 1.0 - ogata_banks(t, v, D, x))
    popt, _ = curve_fit(model, t, c, p0=p0,
                         bounds=([1e-7, 1e-10, 0.005], [1e-1, 1e-4, 0.05]),
                         maxfev=50000)
    pred = model(t, *popt)
    r2 = 1 - np.sum((pred - c)**2) / np.sum((c - c.mean())**2)
    return (*popt, r2)
