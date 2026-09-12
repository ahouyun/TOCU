# Data and External Dependencies

The repository contains TOCU code only. Raw traffic data and runtime-only resources are kept outside the repository.

The training code expects an external project root with the following logical resources:

```text
external-project/
  backbone/
  data-package/

run-root/
  jobs/<backbone>/<dataset>/seed<seed>/
  vendor/<backbone>/              # optional separate runtime copy
```

The exact dataset names used by the current TOCU protocol are `PEMS03`, `PEMS04`, `PEMS08`, and `Seattle`. Data preparation and licensing remain outside this code-only repository.

Do not commit local checkpoint paths, machine-specific job files, raw arrays, or credentials. Keep those resources outside the repository and pass their locations through command-line arguments.
