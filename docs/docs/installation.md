---
icon: lucide/download
---

# Installation

BayesMBAR is written in pure Python and is published on the
[Python Package Index (PyPI)](https://pypi.org/project/bayesmbar/), so you can
install it with virtually any Python package manager. Its dependencies are
installed automatically, with one exception: JAX. Because the right JAX build
depends on your hardware (CPU, GPU, or TPU), we recommend installing JAX first
and then installing BayesMBAR.

## 1. Install JAX

Follow the instructions on the
[JAX website](https://jax.readthedocs.io/en/latest/installation.html) to install
the build that matches your hardware. If you have a GPU available, we strongly
recommend installing the GPU build, as it significantly speeds up the
calculations.

## 2. Install BayesMBAR

Once JAX is installed, add BayesMBAR using your preferred package manager:

=== "pip"

    ```bash
    pip install bayesmbar
    ```

=== "uv"

    ```bash
    # add BayesMBAR to a uv-managed project
    uv add bayesmbar

    # or install it into the active environment
    uv pip install bayesmbar
    ```

That's it. You can verify the installation by importing the package:

```bash
python -c "from bayesmbar import BayesMBAR, FastMBAR; print('BayesMBAR is installed')"
```
