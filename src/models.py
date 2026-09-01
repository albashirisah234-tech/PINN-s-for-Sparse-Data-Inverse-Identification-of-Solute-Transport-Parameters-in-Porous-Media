"""
src/models.py
==============
Neural network architectures used across all forward/inverse PINN
experiments. Pulled into one module so Level 2 (forward, inverse) and
Level 3 (sparsity sweep) all share exactly the same definitions -- earlier
iterations of this project duplicated these classes across scripts, which
is a real risk for silent inconsistency; don't reintroduce that.

Requires: torch (GPU strongly recommended -- see experiments/ scripts for
expected runtimes).
"""
import numpy as np
import torch
import torch.nn as nn


class PINN(nn.Module):
    """Forward PINN: (x, t) -> C. Used when v, D are KNOWN (Level 2a)."""

    def __init__(self, n_hidden=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(2, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers += [nn.Linear(n_hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


class InversePINN(nn.Module):
    """Inverse PINN: (x, t) -> C, with v and D as additional LEARNABLE
    parameters (Level 2b, Level 3). v_init/D_init are the network's starting
    guess for the dimensionless parameters (see non-dimensionalization note
    in experiments/run_level3_sparsity_sweep.py) -- NOT the true values,
    which are unknown to the model by design.

    v and D are parameterized via an unconstrained raw value passed through
    softplus, which keeps them positive throughout training without a hard
    clip that would zero out gradients at the boundary.
    """

    def __init__(self, n_hidden=64, n_layers=4, v_init=1.0, D_init=0.1):
        super().__init__()
        layers = [nn.Linear(2, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers += [nn.Linear(n_hidden, 1)]
        self.net = nn.Sequential(*layers)
        self._v_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(v_init)))))
        self._D_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(D_init)))))

    @property
    def v(self):
        return torch.nn.functional.softplus(self._v_raw)

    @property
    def D(self):
        return torch.nn.functional.softplus(self._D_raw)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


class DataOnlyNN(nn.Module):
    """Plain data-driven baseline: t -> C. Same depth/width as the PINN,
    but no physics loss, no boundary/initial conditions, and critically no
    spatial input -- it can only ever predict at the outlet location where
    real data exists. This last point is deliberate: it is what makes the
    spatial-generalization test (Section 3.6 of the paper) a task only the
    PINN can attempt at all."""

    def __init__(self, n_hidden=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(1, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers += [nn.Linear(n_hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)
