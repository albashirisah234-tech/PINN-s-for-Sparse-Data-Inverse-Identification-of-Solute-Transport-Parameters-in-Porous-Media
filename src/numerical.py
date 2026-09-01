"""
src/numerical.py
==================
Crank-Nicolson finite-difference solver for the 1D ADE, used only as an
independent (non-neural-network) check in Level 1 validation -- confirming
the governing equation, units, and boundary conditions are correctly
specified before any PINN training is attempted.
"""
import numpy as np


def solve_ade_numerical(v, D, L, t_eval, nx=400, C0=1.0):
    """Solve dC/dt = D d2C/dx2 - v dC/dx on x in [0, L].

    Boundary conditions: C(0,t) = C0 (Dirichlet inlet),
                          dC/dx(L,t) = 0 (Neumann outlet, first-order).
    Initial condition:    C(x,0) = 0.

    Returns (t_eval_sorted, C_at_outlet) -- the outlet BTC, matching the
    experimental measurement location.
    """
    dx = L / nx
    dt = min(np.diff(np.sort(t_eval)).min() / 4, 0.5 * dx**2 / D)
    C = np.zeros(nx + 1)
    C[0] = C0

    alpha = D * dt / dx**2
    beta = v * dt / (2 * dx)

    n = nx + 1
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    A[0, 0] = 1.0
    B[0, 0] = 1.0
    for i in range(1, nx):
        A[i, i - 1] = -0.5 * (alpha + beta)
        A[i, i] = 1 + alpha
        A[i, i + 1] = -0.5 * (alpha - beta)
        B[i, i - 1] = 0.5 * (alpha + beta)
        B[i, i] = 1 - alpha
        B[i, i + 1] = 0.5 * (alpha - beta)
    A[nx, nx] = 1.0
    A[nx, nx - 1] = -1.0  # zero-gradient outlet
    B[nx, nx] = 0.0

    Ainv = np.linalg.inv(A)
    t_sorted = np.sort(t_eval)
    out_times, out_vals = [], []
    t_now = 0.0
    ei = 0
    while ei < len(t_sorted):
        rhs = B @ C
        rhs[0] = C0
        rhs[nx] = 0.0
        C = Ainv @ rhs
        t_now += dt
        while ei < len(t_sorted) and t_sorted[ei] <= t_now:
            out_times.append(t_sorted[ei])
            out_vals.append(C[-1])
            ei += 1
    return np.array(out_times), np.array(out_vals)
