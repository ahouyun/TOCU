# Release Scope

This repository is intentionally code-first and contains only the TOCU implementation, tests, audited TOCU snapshots, and reproduction notes.

## Included

```text
README.md
requirements.txt
.gitignore
tocu/__init__.py
tocu/head.py
tocu/history.py
tocu/train.py
tocu/refit.py
tocu/refit_support.py
tocu/audit.py
tocu/audit_seattle.py
tests/test_history.py
tests/test_metrics.py
results/metrics.json
results/audit.json
results/figures/predictive_distributions.png
results/figures/module_ablation.png
docs/data.md
docs/reproducibility.md
docs/third_party.md
docs/repository_scope.md
```

## Excluded

```text
prob_runs/
_quarantine/
work/
output/
.gstack/
preview/
article/
ppt/
pdf/
assets/fonts/
raw datasets
checkpoints and full prediction arrays
SSH, GPU queue, supervisor, and MCP/PPT scripts
paper source and build caches
```

## Naming policy

Public filenames use stable semantic names. They do not expose local experiment dates, revision markers, or historical version labels.

| Local role | Public name |
|---|---|
| probabilistic fusion module | `tocu/head.py` |
| causal history adapter | `tocu/history.py` |
| end-to-end training | `tocu/train.py` |
| formal refit | `tocu/refit.py` |
| refit helpers | `tocu/refit_support.py` |
| integrated audit | `tocu/audit.py` |
| Seattle audit | `tocu/audit_seattle.py` |
| result metrics | `results/metrics.json` |

## Release notes

The repository does not include raw data, runtime-only resources, credentials, or code from other projects. Add a license only after all authors confirm the intended terms.
