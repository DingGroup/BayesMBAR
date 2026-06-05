Overview
========

The MBAR method is very useful for estimating free energy differences between multiple states based on configurations sampled from each state using molecular dynamics simulations or Monte Carlo methods.
The MBAR method can be used as a drop-in replacement for the weighted histogram analysis method (WHAM) and provides extra flexibilities.
BayesMBAR generalizes the MBAR method by providing a Bayesian framework, where the original MBAR method is a special case.
The advantage of the BayesMBAR method is two-fold:

1. It provides more accurate uncertainty estimates for the free energy differences.
2. It provides a principled way to incorporate prior knowledge to improve the accuracy of the free energy estimates.


The BayesMBAR Python package provides an implementation of the BayesMBAR method. In addition, the package includes implementations of `FastMBAR <https://pubs.acs.org/doi/full/10.1021/acs.jctc.8b01010>`_ and `CBayesMBAR <https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c00948>`_.
FastMBAR provides a fast solver for the MBAR equations by taking advantage of GPUs and it is useful when the number of states or the number of configurations is large.
It was originally distributed as the `FastMBAR package <https://fastmbar.readthedocs.io/en/latest/>`_.
CBayesMBAR stands for coupled BayesMBAR. 
It is an example of using the Bayesian framework of BayesMBAR to incorporate prior knowledge to improve the accuracy of the free energy estimates.
It is useful for computing free energy differences and uncertainties on perturbation graphs with cycles and it incorporates the cycle closure constraints to improve the accuracy of the free energy estimates.