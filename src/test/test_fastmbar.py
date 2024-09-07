import pytest
from pytest import approx
import numpy as np
from sys import exit
import math
import os, sys
from bayesmbar import FastMBAR


def test_FastMBAR(setup_data):
    energy, num_conf, F_ref, key = setup_data
    fastmbar = FastMBAR(
        energy,
        num_conf,
        cuda=False,
        cuda_batch_mode=False,
        bootstrap=False,
        verbose=False,
        method="Newton",
    )
    assert fastmbar.F == approx(F_ref, abs = 1e-1)


# @pytest.mark.parametrize("method", ["Newton", "L-BFGS-B"])
# @pytest.mark.parametrize("bootstrap", [False, True])
# def test_FastMBAR_cpus(setup_data, method, bootstrap):
#     energy, num_conf, F_ref = setup_data
#     fastmbar = FastMBAR(
#         energy,
#         num_conf,
#         cuda=False,
#         cuda_batch_mode=False,
#         bootstrap=bootstrap,
#         verbose=False,
#         method=method,
#     )
#     # print(fastmbar.F)
#     # print(F_ref)
#     assert fastmbar.F == approx(F_ref, abs = 1e-1)

# @pytest.mark.skipif(torch.cuda.is_available() is False, reason="CUDA is not avaible")
# @pytest.mark.parametrize("method", ["Newton", "L-BFGS-B"])
# @pytest.mark.parametrize("bootstrap", [False, True])
# @pytest.mark.parametrize("cuda_batch_mode", [False, True])
# def test_FastMBAR_gpus(setup_data, method, bootstrap, cuda_batch_mode):
#     energy, num_conf, F_ref = setup_data
#     fastmbar = FastMBAR(
#         energy,
#         num_conf,
#         cuda=True,
#         cuda_batch_mode=cuda_batch_mode,
#         bootstrap=bootstrap,
#         verbose=False,
#         method=method,
#     )
#     assert fastmbar.F == approx(F_ref, abs = 1e-1)

# energy, num_conf, F_ref = setup_data()
# fastmbar = FastMBAR(
#     energy,
#     num_conf,
#     cuda=False,
#     cuda_batch_mode=False,
#     bootstrap=True,
#     verbose=True,
#     method="Newton",
# )

# res = fastmbar.calculate_free_energies_of_perturbed_states(energy)

# mbar = pymbar.MBAR(energy.numpy(), num_conf.numpy())
# mbar_results = mbar.compute_free_energy_differences(return_theta=True)