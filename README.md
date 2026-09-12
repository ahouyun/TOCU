# TOCU-Topology-Aware-Calibrated-Uncertainty-Modeling-for-Probabilistic-Traffic-Flow-Forecasting

Code and selected result snapshots for the TOCU probabilistic traffic forecasting study.

The repository contains the TOCU uncertainty head, causal history adapter, training and refit entry points, fail-closed result audits, and representative TOCU figures. Raw datasets, runtime-only resources, checkpoints, full prediction arrays, and private infrastructure are intentionally excluded.

## Repository layout

```text
tocu/       Core implementation and command-line entry points
tests/      Lightweight unit tests
results/    Sanitized metrics and representative figures
docs/       Data, reproducibility, and scope notes
```

## Installation

Use Python 3.10 or newer. Install the public dependencies with:

```bash
python -m pip install -r requirements.txt
```

The training path uses locally prepared runtime resources that are intentionally not included in this repository.

## Quick checks

From the repository root:

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q tocu tests
```

## Training

The training entry point accepts local runtime and run directories containing the prepared configuration and frozen point-prediction checkpoint:

```bash
python -m tocu.train \
  --root /path/to/external-project \
  --run-root /path/to/run-root \
  --dataset PEMS08 \
  --seed 0 \
  --output runs/tocu/PEMS08/seed0
```

Use `--help` to inspect the complete set of TOCU training and ablation options. Runtime artifacts are written to the local output directory supplied by the user.

The refit entry point provides the TOCU calibration refitting workflow:

```bash
python -m tocu.refit --help
```

## Auditing results

For a complete four-dataset run with seeds `0`, `1`, and `2`:

```bash
python -m tocu.audit \
  --root runs/tocu \
  --output results/audit.json
```

For the standalone Seattle formal run:

```bash
python -m tocu.audit_seattle \
  --root runs/tocu/seattle-formal \
  --output results/seattle-audit.json
```

The auditors verify artifact completeness, target hashes, metric recomputation, split configuration, innovation flags, and the validation-only test-read policy.

## Result snapshot

`results/metrics.json` records the audited three-seed mean and sample standard deviation for the four datasets. The values are provided as a provenance snapshot; they do not replace a fresh run on a newly prepared environment.

The result files contain TOCU-only audited snapshots and do not include comparison-model code or metrics.

## License

No license is included yet. Add one after all authors confirm the intended terms.
