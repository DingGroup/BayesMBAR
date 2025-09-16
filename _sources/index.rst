.. BayesMBAR documentation master file, created by
   sphinx-quickstart on Fri Mar  8 13:56:32 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BayesMBAR!
=====================

BayesMBAR stands for `Bayesian Multistate Bennett Acceptance Ratio methods <https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c01212>`_. 
It is a Bayesian generalization of the `Multistate Bennett Acceptance Ratio (MBAR) method <https://pubs.aip.org/aip/jcp/article/129/12/124105/957527>`_.
It is useful for computing free energy differences and uncertainties between multiple states based on configurations sampled from each state using molecular dynamics simulations or Monte Carlo methods.
It can be used as a drop-in replacement for the weighted histogram analysis method (WHAM) and provides extra flexibilities such as the ability to compute uncertainties and to incorporate prior knowledge to improve the accuracy of the free energy estimates.
Besides the BayesMBAR method, the package also includes implementations of `FastMBAR <https://pubs.acs.org/doi/full/10.1021/acs.jctc.8b01010>`_ and `CBayesMBAR <https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c00948>`_.
All the methods are implemented using `JAX <https://jax.readthedocs.io/en/latest/>`_ and can be run on CPUs, GPUs, and TPUs.




.. toctree::
   :hidden:
   :maxdepth: 2
   
   overview
   installation
   examples

.. toctree::
   :hidden:
   :maxdepth: 2
   
   API