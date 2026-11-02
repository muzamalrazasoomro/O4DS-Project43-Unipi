# Project 43 — Convex Quadratic Min-Cost Flow
University of Pisa, Optimization for Data Science.

## Run
```bash
python -m pip install -r requirements.txt
python run_experiments.py
```
Optional: `python run_experiments.py --nodes 50 --arcs 150 --iterations 10000`

Results are written to `results/`: `summary.csv`, histories, and plots.

A1 is a custom smoothed dual method (standard and accelerated).
A2 is a CVXPY/OSQP benchmark. Instances are connected and guaranteed feasible.

**Interpretation:** `best_dual_lb` is a valid lower bound. The generator's
known feasible flow gives `certified_upper_bound`. The difference is a
certified (possibly loose) gap. The averaged objective is *not* a certified
upper bound while its conservation residual is nonzero. A2 is a numerical
reference, not A1's own upper bound.

The suggested smoothing parameter uses an empirical pilot estimate of
dual distance, not a mathematically certified radius. The exact averaging
identity in the report applies to standard gradient ascent, not acceleration.
