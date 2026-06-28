---
icon: lucide/home
---

# Welcome to BayesMBAR!

BayesMBAR stands for Bayesian Multistate Bennett Acceptance Ratio methods.[^bayesmbar]<sup>,</sup>[^cbayesmbar]
It is a Bayesian generalization of the Multistate Bennett Acceptance Ratio (MBAR) method.[^mbar]<sup>,</sup>[^geyer]

MBAR is useful for computing free energy differences among multiple thermodynamic states using
configurations sampled from each state.
The sampling can be performed using any method that can generate configurations according to the Boltzmann distribution, such as molecular dynamics simulations or Monte Carlo methods.
MBAR can be used as a drop-in replacement for the weighted histogram analysis method (WHAM)[^wham] and provides extra flexibilities.
It is widely used in the field of computational chemistry and physics for calculating free energies.
Example applications include computing the potential of mean force along a reaction coordinate, calculating hydration free energies of small molecules, and estimating binding free energies of ligands to proteins.

BayesMBAR generalizes the MBAR method by providing a Bayesian framework, where the original MBAR method
is a special case.
The advantage of the BayesMBAR method is two-fold:

1. It provides more accurate uncertainty estimates for the free energy differences.
2. It provides a principled way to incorporate prior knowledge to improve the accuracy of the free
   energy estimates. When the prior is chosen to be non-informative, BayesMBAR reduces to MBAR.


<!-- Besides the BayesMBAR method, the package also includes implementations of
FastMBAR[^fastmbar] and CBayesMBAR.[^cbayesmbar]
All the methods are implemented using [JAX](https://jax.readthedocs.io/en/latest/) and can be run on
CPUs, GPUs, and TPUs. -->



<div class="grid cards" markdown>

<!-- - :material-book-open-variant: **[Overview](overview.md)** — what BayesMBAR is and why to use it -->
- :material-download: **[Installation](installation.md)**
- :material-book-open-variant: **[Usage](usage.md)**
- :material-flask: **[Examples](examples/bayesmbar_for_harmonic_oscillators.md)**
- :material-api: **[API reference](api.md)**
- :material-frequently-asked-questions: **[FAQ](faq.md)**

</div>

## Citation

If you use BayesMBAR in your research, please cite the BayesMBAR paper.[^bayesmbar]<sup>,</sup>[^cbayesmbar]

[^bayesmbar]:
    Ding, X. Bayesian Multistate Bennett Acceptance Ratio Methods.
    *J. Chem. Theory Comput.* <https://doi.org/10.1021/acs.jctc.3c01212>

[^cbayesmbar]:
    Coupled Bayesian Multistate Bennett Acceptance Ratio (CBayesMBAR).
    *J. Chem. Theory Comput.* <https://doi.org/10.1021/acs.jctc.4c00948>

[^mbar]:
    Shirts, M. R.; Chodera, J. D. Statistically Optimal Analysis of Samples
    from Multiple Equilibrium States. *J. Chem. Phys.*
    <https://doi.org/10.1063/1.2978177>

[^geyer]:
    Geyer, C. J. Estimating Normalizing Constants and Reweighting Mixtures;
    Report; 1994.

[^wham]:
    Kumar, S.; Rosenberg, J. M.; Bouzida, D.; Swendsen, R. H.; Kollman, P. A.
    The Weighted Histogram Analysis Method for Free-Energy Calculations on
    Biomolecules. I. The Method. *J. Comput. Chem.* 1992, *13*, 1011–1021.
    <https://doi.org/10.1002/jcc.540130812>
