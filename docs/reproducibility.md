# Reproducibility Protocol

The current audited protocol uses:

- datasets: `PEMS03`, `PEMS04`, `PEMS08`, and `Seattle`;
- three random seeds: `0`, `1`, and `2`;
- input length: `12`;
- forecast length: `12`;
- split: `0.6 / 0.2 / 0.2` for train, validation, and test;
- 50 distribution samples for the probabilistic summary;
- validation-only selection, with test labels consumed only by the final evaluation step.

The uncertainty head combines a low-rank structural component with an orthogonal residual component. Its scale is conditioned on frequency residual energy and topology-aware impedance risk. The formal audit also checks the three recorded innovation flags:

```text
frequency_conditioned
orthogonal_two_source
frequency_impedance_variance_consistency
```

For a release-quality reproduction, record the runtime environment identifier, dataset checksums, Python version, PyTorch version, device, command line, and resulting target hashes alongside the audit JSON. Those environment-specific records are intentionally kept outside this repository.
