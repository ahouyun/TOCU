# Data and External Dependencies

The public repository does not include raw traffic data or third-party model repositories.

The training code expects an external project root with the following logical resources:

```text
external-project/
  repos/<upstream-model>/
  deps/<upstream-data-package>/

run-root/
  jobs/<upstream-model>/<dataset>/seed<seed>/
  vendor/<upstream-model>/        # used when the dataset needs a separate copy
```

The exact dataset names used by the current protocol are `PEMS03`, `PEMS04`, `PEMS08`, and `Seattle`. Data preparation, licensing, and redistribution conditions remain the responsibility of the data and upstream-model providers.

Do not commit local checkpoint paths, machine-specific job files, raw arrays, or credentials. Keep those resources outside the repository and pass their locations through command-line arguments.
