# Third-Party Scope

The repository was organized after reviewing common layouts in public research repositories for traffic forecasting and probabilistic time-series modeling.

The shared practices adopted here are:

- one clear README with installation and execution commands;
- a small, named implementation package;
- explicit test and audit entry points;
- a dependency file;
- a data and reproducibility note;
- representative result snapshots instead of complete experiment histories.

The following are deliberately out of scope for the first release:

- copied third-party source code whose license or redistribution terms are not confirmed;
- raw datasets and unverified data variants;
- model checkpoints and full prediction arrays;
- remote GPU, SSH, queue, supervisor, or synchronization scripts;
- paper source, build caches, slides, and internal analysis tools.

The upstream implementation and its license must be installed and reviewed separately before redistribution or bundling.
