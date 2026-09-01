# Physics-Informed Neural Networks for Sparse-Data Inverse Identification of Solute Transport Parameters in Porous Media

Code accompanying the paper of the same name. Investigates whether
physics-informed neural networks (PINNs) can recover advection-dispersion
equation (ADE) parameters — pore velocity `v` and dispersion coefficient
`D` — from sparse breakthrough-curve (BTC) observations, and characterizes
how observation density, random initialization, and transport regime
(loading vs. unloading) affect recovery accuracy.

## Upload checklist (read this first if uploading manually to GitHub)

This is the complete, final, verified file list — 25 files total. Every
number in the accompanying paper was checked against these exact files
before this list was written. If you unzip the provided archive and
upload its contents as-is, this checklist is satisfied automatically —
you do not need to hand-pick files from other folders or downloads.

```
your-repo-root/
├── README.md                              (this file)
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md                          (placeholder only — do NOT upload the actual .xlsx, see "Data" below)
├── src/
│   ├── __init__.py
│   ├── data_io.py
│   ├── analytical.py
│   ├── numerical.py
│   ├── models.py
│   ├── physics.py
│   └── training.py
├── experiments/
│   ├── run_level1_validation.py
│   ├── run_level2_pinn.py
│   ├── run_level3_sparsity_sweep.py
│   └── make_paper_figures.py
└── results/
    ├── tables/
    │   ├── ground_truth_parameters.csv
    │   ├── level3_sparsity_results_raw.csv
    │   └── table3_sparsity_hierarchical.csv
    └── figures/
        ├── fig1_conceptual.png                    (Figure 1 in the paper)
        ├── fig2_level1_validation.png              (Figure 2)
        ├── fig3_forward_pinn_validation.png        (Figure 3)
        ├── fig4_inverse_pinn_baseline.png          (Figure 4)
        ├── fig5_sparsity_pooled.png                (Figure 5)
        ├── fig6_seed_bimodality.png                (Figure 6)
        └── fig7_load_vs_unload.png                 (Figure 7)
```

**Note on fig3 and fig4**: these two were produced on Kaggle GPU (this
development environment has no GPU/PyTorch), by `run_level2_pinn.py
--mode forward` and `--mode inverse` respectively. They are included
here as the actual output files from that run, not regenerated copies —
re-running `run_level2_pinn.py` yourself will reproduce them (the script
saves under its own default filenames; rename to match the paper's
figure numbering if you want the correspondence to stay obvious).

The manuscript itself (`.docx`) is delivered separately, not inside this
archive — it isn't code or data. If you want it version-controlled
alongside the code, create a `paper/` folder in your repo and add it
there yourself; this is optional and not required for reproducibility.

## Data

Experimental breakthrough curves are from a published micromodel
dye-tracer study:

> Erfani, H., Karadimitriou, N., Nissan, A., Walczak, M. S., An, S.,
> Berkowitz, B., & Niasar, V. (2021). Process-dependent solute transport
> in porous media. *Transport in Porous Media*, 140(1), 421–435.
> https://doi.org/10.1007/s11242-021-01655-6

Dataset (CC BY 4.0):

> Erfani Gahrooei, H., & Niasar, V. (2021). Process-Dependent Solute
> Transport in Porous Media [Data set]. Mendeley Data, V1.
> https://doi.org/10.17632/gzf44vc7cr.1

Download `Homogeneous_MoM_BTC.xlsx` from the Mendeley link above and place
it in `data/` before running anything. **Note the source filename has a
space in it (`Homogeneous MoM_BTC.xlsx`) — rename to
`Homogeneous_MoM_BTC.xlsx` (underscore) to match what the scripts expect,
or pass `--data` with the exact path/filename you used.**

This project uses only the 10 *homogeneous* micromodel curves
(Experiments 1–10). The heterogeneous (fine-coarse) curves in the same
dataset are not yet analyzed with this pipeline (see Limitations in the
paper).

## Setup

```bash
pip install -r requirements.txt
```

A GPU is strongly recommended for `run_level2_pinn.py` and especially
`run_level3_sparsity_sweep.py` (the full sweep is ~180 training runs,
~3.4 hours on a single GPU; CPU-only would be substantially slower).
`run_level1_validation.py` does not use PyTorch and runs in seconds on
CPU.

## Project structure

```
src/                          Shared, reusable modules
  data_io.py                  Loading BTC data, sparsity subsampling
  analytical.py               Ogata-Banks solution, ground-truth fitting
  numerical.py                Crank-Nicolson finite-difference solver
  models.py                   PINN, InversePINN, DataOnlyNN (torch.nn.Module)
  physics.py                  PDE residual (autograd)
  training.py                 Training loops for all three model types

experiments/                  Thin runner scripts (argparse-configurable)
  run_level1_validation.py    Ground-truth table + numerical/analytical/data validation
  run_level2_pinn.py          Single-experiment forward and inverse PINN
  run_level3_sparsity_sweep.py  Full multi-seed sparsity sweep (the main result)
  make_paper_figures.py       Reproduces the paper's summary figures + stats tests

data/                         Place Homogeneous_MoM_BTC.xlsx here (not tracked in git)
results/tables/               Output CSVs (ground truth, sparsity sweep results)
results/figures/              Output PNGs
```

## Reproducing the results

```bash
# Level 1: ground-truth parameter table + numerical solver validation
python experiments/run_level1_validation.py --data data/Homogeneous_MoM_BTC.xlsx

# Level 2: forward PINN validation (full (x,t) field vs. analytical solution)
python experiments/run_level2_pinn.py --data data/Homogeneous_MoM_BTC.xlsx --mode forward

# Level 2: inverse PINN, single curve, full resolution
python experiments/run_level2_pinn.py --data data/Homogeneous_MoM_BTC.xlsx --mode inverse

# Level 3: smoke test first (Exp 7 only, ~15-20 min) -- always do this before the full sweep
python experiments/run_level3_sparsity_sweep.py --data data/Homogeneous_MoM_BTC.xlsx --smoke-test

# Level 3: full sweep (10 experiments x 6 sparsity levels x 3 seeds, ~3.4 hours on GPU)
python experiments/run_level3_sparsity_sweep.py --data data/Homogeneous_MoM_BTC.xlsx

# Reproduce the paper's summary figures and statistical tests from the Level 3 output
python experiments/make_paper_figures.py --raw results/tables/level3_sparsity_results_raw.csv
```

**Important**: if running in an ephemeral environment (e.g. a Kaggle
notebook), download the contents of `results/` immediately after the
Level 3 sweep finishes, before doing anything else — some environments do
not reliably persist working directories between sessions, and losing 3+
hours of compute output is expensive to recover from (it is possible to
reconstruct from console output text if you saved it, but don't rely on
needing to).

## Key results summary

| Finding | Where |
|---|---|
| Forward PINN reproduces analytical solution, RMSE = 0.0143 over full (x,t) field | `run_level2_pinn.py --mode forward` |
| Velocity recovered far more reliably than dispersion (single-digit % vs. 10-90%+ error) | Level 3 sweep, all experiments; `table3_sparsity_hierarchical.csv` |
| Dispersion recovery is bimodal across random seeds; 9/10 experiments show >10pp seed-to-seed spread at full resolution | `make_paper_figures.py` -> `fig6_seed_bimodality.png`, `seed_variability_summary()` |
| Lowest-training-loss seed selection is unreliable: 29/60 (48%), binomial test p=0.90 vs. chance | `make_paper_figures.py` -> `best_of_k_reliability_check()` |
| Loading curves recovered far more reliably than unloading curves (Mann-Whitney U=0, p=0.004) | `make_paper_figures.py` -> `fig7_load_vs_unload.png` |
| Spatial-generalization trend seen in one illustrative experiment does NOT hold up when tested across all 10 (Wilcoxon p=0.125, Spearman p=0.28) -- reported as an open question, not a finding | `make_paper_figures.py` -> `spatial_generalization_significance_test()` |

Every number above was regenerated from `results/tables/level3_sparsity_results_raw.csv` and
matched exactly against the submitted manuscript before this repository was finalized --
`make_paper_figures.py` is not a description of the analysis, it IS the analysis. Run it
yourself against the raw CSV to verify any number in the paper rather than taking it on trust.

See the accompanying paper for full methodology, discussion, and
comparison against related work.

## Methodological notes worth knowing before modifying this code

- **Units** (`src/analytical.py`): the source spreadsheet has no stated
  units for x, t, v, D. These were recovered empirically by fitting the
  analytical solution with sensor location `x` free and checking it
  converges near the paper's stated 3 cm micromodel length. Don't assume
  different units without re-deriving this.
- **Ground truth uses a FIXED sensor length** (`L_KNOWN = 0.03` in
  `src/analytical.py`), not the floating-`x` fit above. This is
  deliberate: the inverse PINN cannot know the true sensor length without
  leaking the answer into its own setup, so it also assumes a fixed
  length — and ground truth must use the same assumption for a fair
  comparison. Mixing the two conventions when comparing numbers will
  produce misleading errors (this happened once during development; see
  Section 2.5 of the paper for the full explanation).
- **Loss weights** (`src/training.py`) were tuned once and then held fixed
  across every experiment and sparsity level, by design — re-tuning them
  per condition would confound the sparsity comparison.
- **Best-of-K is not a working selection protocol** — see Section 3.5 of
  the paper. It's included here to reproduce that specific finding
  (training loss does not predict accuracy), not as a recommended way to
  pick a final model.
