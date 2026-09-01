"""
src/data_io.py
================
Loading experimental breakthrough-curve (BTC) data and building sparse
observation subsets from it.

Data source: Erfani et al. (2021), "Process-dependent solute transport in
porous media," Transport in Porous Media, 140(1), 421-435.
https://doi.org/10.1007/s11242-021-01655-6
Dataset: Erfani Gahrooei & Niasar (2021), Mendeley Data, V1.
https://doi.org/10.17632/gzf44vc7cr.1

Expected input file: Homogeneous_MoM_BTC.xlsx, with one sheet per experiment
("Exp 1" .. "Exp 10"), each sheet a two-column (time, C/C0) table.
"""
import numpy as np
import openpyxl

# Odd-numbered experiments are loading curves (C rises 0 -> C0);
# even-numbered are unloading curves (C falls C0 -> 0). This is a property
# of how the source dataset pairs each flow rate's two runs, not something
# inferred from the data itself.
KIND_OF = {f"Exp {i}": ("load" if i % 2 == 1 else "unload") for i in range(1, 11)}
EXPERIMENTS = [f"Exp {i}" for i in range(1, 11)]


def load_curve(sheet_name, path):
    """Load one experiment's (time, C/C0) series from the source spreadsheet.

    Returns (t, c) as numpy arrays. Only rows where both columns are
    numeric are kept, which naturally skips header/label rows.
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet_name]
    t, c = [], []
    for row in ws.iter_rows(min_row=1, values_only=True):
        if isinstance(row[0], (int, float)) and isinstance(row[1], (int, float)):
            t.append(row[0])
            c.append(row[1])
    return np.array(t), np.array(c)


def subsample_fixed_count(t, c, n_points):
    """Evenly-spaced-by-index subsample of n_points from a full curve.

    Using a FIXED POINT COUNT (rather than a fixed percentage) matters here
    because curve lengths differ substantially across experiments (551 to
    3,305 samples) -- a fixed percentage would not be comparable across
    experiments, whereas a fixed count is.

    n_points=None returns the full curve unchanged.
    """
    if n_points is None or n_points >= len(t):
        return t.copy(), c.copy()
    idx = np.unique(np.linspace(0, len(t) - 1, n_points).round().astype(int))
    return t[idx], c[idx]
