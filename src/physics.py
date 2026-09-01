"""
src/physics.py
================
The PDE residual for the 1D advection-dispersion equation, computed via
automatic differentiation. Used identically by the forward PINN (with
fixed v, D) and the inverse PINN (with learnable v, D) -- the only
difference between forward and inverse training is whether v/D come from
a constant or from model.v / model.D (see src/models.py).
"""
import torch


def pde_residual(model, x, t, D, v):
    """Residual of  dC/dt - D*d2C/dx2 + v*dC/dx = 0  at points (x, t).

    D and v can be plain floats/tensors (forward problem, fixed) or
    model.D / model.v properties (inverse problem, learnable) -- this
    function does not care which, by design.
    """
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    C = model(x, t)
    C_t = torch.autograd.grad(C, t, torch.ones_like(C), create_graph=True)[0]
    C_x = torch.autograd.grad(C, x, torch.ones_like(C), create_graph=True)[0]
    C_xx = torch.autograd.grad(C_x, x, torch.ones_like(C_x), create_graph=True)[0]
    return C_t - D * C_xx + v * C_x
