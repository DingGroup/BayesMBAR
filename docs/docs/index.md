---
icon: lucide/home
---

# Welcome to BayesMBAR

BayesMBAR stands for [Bayesian Multistate Bennett Acceptance Ratio methods](https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c01212).
It is a Bayesian generalization of the [Multistate Bennett Acceptance Ratio (MBAR) method](https://pubs.aip.org/aip/jcp/article/129/12/124105/957527).
It is useful for computing free energy differences and uncertainties between multiple states based on
configurations sampled from each state using molecular dynamics simulations or Monte Carlo methods.
It can be used as a drop-in replacement for the weighted histogram analysis method (WHAM) and provides
extra flexibilities such as the ability to compute uncertainties and to incorporate prior knowledge to
improve the accuracy of the free energy estimates.

Besides the BayesMBAR method, the package also includes implementations of
[FastMBAR](https://pubs.acs.org/doi/full/10.1021/acs.jctc.8b01010) and
[CBayesMBAR](https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c00948).
All the methods are implemented using [JAX](https://jax.readthedocs.io/en/latest/) and can be run on
CPUs, GPUs, and TPUs.

## Where to go next

<div class="grid cards" markdown>

- :material-book-open-variant: **[Overview](overview.md)** — what BayesMBAR is and why to use it
- :material-download: **[Installation](installation.md)** — install BayesMBAR with pip
- :material-flask: **[Examples](examples/bayesmbar_for_harmonic_oscillators.md)** — worked examples and tutorials
- :material-api: **[API reference](api.md)** — the full Python API
- :material-frequently-asked-questions: **[FAQ](faq.md)** — frequently asked questions

</div>

## Citation

If you use BayesMBAR in your research, please cite the
[BayesMBAR paper](https://doi.org/10.1021/acs.jctc.3c01212).
