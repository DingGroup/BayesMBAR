Installation
============

BayesMBAR is written in pure Python and can be installed using the pip package manager.
BayesMBAR depends on several Python packages.
Although many of these packages will be installed automatically as dependencies when you install BayesMBAR with pip, one of the dependencies, JAX, needs to be installed separately.
We suggest following these steps to install BayesMBAR:

1. Installing JAX
-----------------
Follow the instructions on the `JAX website <https://jax.readthedocs.io/en/latest/installation.html#>`_ to install JAX.
We highly recommend using the GPU version of JAX if you have a GPU available, as it will significantly speed up the calculations.

2. Installing BayesMBAR
------------------------
Once JAX is installed, you can install BayesMBAR using pip:

.. code-block:: bash

  pip install bayesmbar



