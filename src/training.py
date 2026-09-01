"""
src/training.py
=================
Training loops for the forward PINN, inverse PINN, and data-only NN
baseline. Kept separate from src/models.py so the model definitions stay
easy to scan on their own.

All loss weights (10x on IC/BC, 50x on data) were tuned once during initial
development and then held fixed across every experiment in this project --
they are not re-tuned per experiment/sparsity level. This is a deliberate
methodological choice for a fair sparsity comparison, not an oversight.
"""
import numpy as np
import torch
from .models import PINN, InversePINN, DataOnlyNN
from .physics import pde_residual

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_bc_ic_points(N_COLLOC, N_IC, N_BC, t_max_dimless=1.0):
    x_f = torch.rand(N_COLLOC, 1)
    t_f = torch.rand(N_COLLOC, 1) * t_max_dimless
    x_ic = torch.rand(N_IC, 1)
    t_ic = torch.zeros(N_IC, 1)
    x_bc_in = torch.zeros(N_BC, 1)
    t_bc = torch.rand(N_BC, 1) * t_max_dimless
    x_bc_out = torch.ones(N_BC, 1)
    t_bc_out = torch.rand(N_BC, 1) * t_max_dimless
    return x_f, t_f, x_ic, t_ic, x_bc_in, t_bc, x_bc_out, t_bc_out


def train_forward_pinn(v_true_dimless, D_true_dimless, n_epochs=8000, lr=1e-3,
                        t_max_dimless=4.0, log_every=500, seed=0):
    """Level 2a: solve the forward ADE with v, D KNOWN (no observed data
    used at all -- purely physics + boundary/initial conditions)."""
    torch.manual_seed(seed)
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N_COLLOC, N_IC, N_BC = 4000, 400, 400
    x_f, t_f, x_ic, t_ic, x_bc_in, t_bc, x_bc_out, t_bc_out = \
        [z.to(device) for z in _make_bc_ic_points(N_COLLOC, N_IC, N_BC, t_max_dimless)]
    c_ic = torch.zeros(N_IC, 1, device=device)
    c_bc_in = torch.ones(N_BC, 1, device=device)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss_pde = torch.mean(pde_residual(model, x_f, t_f, D_true_dimless, v_true_dimless) ** 2)
        loss_ic = torch.mean((model(x_ic, t_ic) - c_ic) ** 2)
        loss_bc_in = torch.mean((model(x_bc_in, t_bc) - c_bc_in) ** 2)

        x_out = x_bc_out.clone().requires_grad_(True)
        C_out = model(x_out, t_bc_out)
        C_out_x = torch.autograd.grad(C_out, x_out, torch.ones_like(C_out), create_graph=True)[0]
        loss_bc_out = torch.mean(C_out_x ** 2)

        loss = loss_pde + 10 * loss_ic + 10 * loss_bc_in + loss_bc_out
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0:
            print(f"  epoch {epoch:5d}  loss={loss.item():.3e}")
    return model


def train_inverse_pinn(t_obs_nondim, c_obs, kind, seed=0, n_epochs=10000, lr=1e-3,
                        v_init=1.0, D_init=0.1):
    """Level 2b / Level 3: v and D UNKNOWN, learned jointly with the network
    from the observed (possibly sparse) outlet curve. Returns (model,
    final_loss) -- final_loss is used by experiments/run_level3... for the
    best-of-K analysis (Section 3.5 of the paper)."""
    torch.manual_seed(seed)
    model = InversePINN(v_init=v_init, D_init=D_init).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_data = torch.ones(len(t_obs_nondim), 1)
    t_data = torch.tensor(t_obs_nondim, dtype=torch.float32).unsqueeze(1)
    c_data = torch.tensor(c_obs, dtype=torch.float32).unsqueeze(1)

    N_COLLOC, N_IC, N_BC = 4000, 400, 400
    x_f, t_f, x_ic, t_ic, x_bc_in, t_bc, x_bc_out, t_bc_out = \
        _make_bc_ic_points(N_COLLOC, N_IC, N_BC, t_max_dimless=1.0)

    # unloading curves have IC/BC values flipped relative to loading
    c_ic = torch.zeros(N_IC, 1) if kind == 'load' else torch.ones(N_IC, 1)
    c_bc_in = torch.ones(N_BC, 1) if kind == 'load' else torch.zeros(N_BC, 1)

    tens = [x_data, t_data, c_data, x_f, t_f, x_ic, t_ic, c_ic,
            x_bc_in, t_bc, c_bc_in, x_bc_out, t_bc_out]
    (x_data, t_data, c_data, x_f, t_f, x_ic, t_ic, c_ic,
     x_bc_in, t_bc, c_bc_in, x_bc_out, t_bc_out) = [z.to(device) for z in tens]

    final_loss = None
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss_pde = torch.mean(pde_residual(model, x_f, t_f, model.D, model.v) ** 2)
        loss_ic = torch.mean((model(x_ic, t_ic) - c_ic) ** 2)
        loss_bc_in = torch.mean((model(x_bc_in, t_bc) - c_bc_in) ** 2)

        x_out = x_bc_out.clone().requires_grad_(True)
        C_out = model(x_out, t_bc_out)
        C_out_x = torch.autograd.grad(C_out, x_out, torch.ones_like(C_out), create_graph=True)[0]
        loss_bc_out = torch.mean(C_out_x ** 2)

        loss_data = torch.mean((model(x_data, t_data) - c_data) ** 2)

        loss = loss_pde + 10 * loss_ic + 10 * loss_bc_in + loss_bc_out + 50 * loss_data
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    return model, final_loss


def train_data_only_nn(t_obs_nondim, c_obs, seed=0, n_epochs=10000, lr=1e-3):
    """Baseline: plain supervised fit to the sparse outlet points ONLY --
    no physics, no boundary/initial conditions, no spatial input."""
    torch.manual_seed(seed)
    model = DataOnlyNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    t_data = torch.tensor(t_obs_nondim, dtype=torch.float32).unsqueeze(1).to(device)
    c_data = torch.tensor(c_obs, dtype=torch.float32).unsqueeze(1).to(device)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = torch.mean((model(t_data) - c_data) ** 2)
        loss.backward()
        optimizer.step()
    return model
