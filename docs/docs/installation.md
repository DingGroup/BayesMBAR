---
icon: lucide/download
---

# Installation

BayesMBAR is written in pure Python and is published on the
[Python Package Index (PyPI)](https://pypi.org/project/bayesmbar/), so you can
install it with virtually any Python package manager. All of its dependencies,
including JAX, are installed automatically. By default this gives you the CPU
build of JAX; if you have a GPU available, we strongly recommend using a GPU
build, as it significantly speeds up the calculations.

## 1. CPU only

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

## 2. NVIDIA GPU (Linux)

Install the `cuda` extra, which pulls in the CUDA 12 build of JAX:

=== "pip"

    ```bash
    pip install "bayesmbar[cuda]"
    ```

=== "uv"

    ```bash
    # add BayesMBAR to a uv-managed project
    uv add "bayesmbar[cuda]"

    # or install it into the active environment
    uv pip install "bayesmbar[cuda]"
    ```

## 3. Other hardware

For other accelerators (e.g., TPUs, AMD GPUs, or a different CUDA version),
first install the matching JAX build by following the instructions on the
[JAX website](https://jax.readthedocs.io/en/latest/installation.html), then
install BayesMBAR as in the CPU-only case above. The JAX build you installed
will be kept.

That's it. You can verify the installation by importing the package:

```bash
python -c "from bayesmbar import BayesMBAR, FastMBAR; print('BayesMBAR is installed')"
```
