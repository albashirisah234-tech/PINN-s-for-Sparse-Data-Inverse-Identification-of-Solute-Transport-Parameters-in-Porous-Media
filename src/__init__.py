"""
src/ -- shared modules for the PINN solute-transport project.

    data_io      loading BTC data, sparsity subsampling
    analytical   Ogata-Banks analytical solution, ground-truth fitting
    numerical    Crank-Nicolson finite-difference solver (Level 1 only)
    models       PINN, InversePINN, DataOnlyNN (torch.nn.Module classes)
    physics      PDE residual (shared by forward and inverse training)
    training     training loops for all three model types
"""
