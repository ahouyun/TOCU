# Release Scope

This is the first private-repository staging scope. It is intentionally code-first and small enough to review before the GitHub repository is created.

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
| selection record | `selection.json` |
| training checkpoint output | `checkpoint.pt` |
| distribution summary output | `summary.npz` |

## Before pushing

1. Confirm the GitHub owner and repository name.
2. Confirm that all authors approve the private repository scope.
3. Confirm the upstream and dataset licensing position.
4. Decide whether to add `MIT`, `Apache-2.0`, or another license.
5. Re-run the static credential/path scan and unit tests from this directory.
6. Initialize Git and push only after the staging tree is reviewed.
