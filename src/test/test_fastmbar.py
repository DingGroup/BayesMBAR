import numpy as np
import pytest
from pytest import approx
from bayesmbar import FastMBAR


@pytest.mark.parametrize("method", ["Newton", "L-BFGS"])
@pytest.mark.parametrize("bootstrap", [False, True])
def test_FastMBAR(setup_mbar_data, method, bootstrap):
    energy, num_conf, F_ref, energy_p, F_ref_p = setup_mbar_data
    fastmbar = FastMBAR(
        energy,
        num_conf,
        bootstrap=bootstrap,
        bootstrap_num_rep = 30,
        verbose=False,
        method=method,
    )
    assert fastmbar.F == approx(F_ref, abs = 2e-1)

    results = fastmbar.calculate_free_energies_of_perturbed_states(energy_p)
    results['F'] = results['F'] - results['F'].mean()
    assert results['F'] == approx(F_ref_p, abs = 2e-1)

@pytest.mark.parametrize("method", ["Newton", "L-BFGS"])
@pytest.mark.parametrize("batch_size", [7, 50, 100_000])
def test_FastMBAR_batched_matches_unbatched(setup_mbar_data, method, batch_size):
    energy, num_conf, F_ref, energy_p, F_ref_p = setup_mbar_data
    ref = FastMBAR(energy, num_conf, verbose=False, method=method)
    fastmbar = FastMBAR(
        energy, num_conf, verbose=False, method=method, batch_size=batch_size
    )
    assert fastmbar.F == approx(ref.F, abs=1e-10)
    assert fastmbar.F_cov == approx(ref.F_cov, abs=1e-10)
    assert fastmbar.log_prob_mix == approx(ref.log_prob_mix, abs=1e-10)

    results = fastmbar.calculate_free_energies_of_perturbed_states(energy_p)
    results_ref = ref.calculate_free_energies_of_perturbed_states(energy_p)
    assert results["F"] == approx(results_ref["F"], abs=1e-10)
    assert results["F_std"] == approx(results_ref["F_std"], abs=1e-10)


@pytest.mark.parametrize("method", ["Newton", "L-BFGS"])
@pytest.mark.parametrize("batch_size", [7, 100_000])
def test_FastMBAR_batched_bootstrap_matches_unbatched(
    setup_mbar_data, monkeypatch, method, batch_size
):
    energy, num_conf, F_ref, energy_p, F_ref_p = setup_mbar_data
    ## the bootstrap is seeded from the clock; pin it so that both runs draw the
    ## same resamples
    monkeypatch.setattr("bayesmbar.fastmbar.time", lambda: 1234.5)

    kwargs = dict(bootstrap=True, bootstrap_num_rep=5, verbose=False, method=method)
    ref = FastMBAR(energy, num_conf, **kwargs)
    fastmbar = FastMBAR(energy, num_conf, batch_size=batch_size, **kwargs)
    assert fastmbar.F == approx(ref.F, abs=1e-10)
    assert fastmbar.F_std == approx(ref.F_std, abs=1e-10)
    assert fastmbar.log_prob_mix == approx(ref.log_prob_mix, abs=1e-10)

    results = fastmbar.calculate_free_energies_of_perturbed_states(energy_p)
    results_ref = ref.calculate_free_energies_of_perturbed_states(energy_p)
    assert results["F"] == approx(results_ref["F"], abs=1e-10)
    assert results["F_std"] == approx(results_ref["F_std"], abs=1e-10)


@pytest.mark.parametrize("bootstrap", [False, True])
def test_FastMBAR_perturbed_states_equal_to_sampled_states(setup_mbar_data, bootstrap):
    ## perturbed states identical to the sampled states must reproduce their
    ## free energies, including the additive constant
    energy, num_conf, F_ref, energy_p, F_ref_p = setup_mbar_data
    fastmbar = FastMBAR(
        energy, num_conf, bootstrap=bootstrap, bootstrap_num_rep=10, verbose=False
    )
    results = fastmbar.calculate_free_energies_of_perturbed_states(energy)
    assert results["F"] == approx(fastmbar.F, abs=1e-4)


## Streamed Newton takes the same iterates as the unbatched Newton solver, so its
## results agree to rounding. Streamed L-BFGS uses SciPy's L-BFGS-B, which reaches
## the same optimum along a different path, so it agrees to the convergence
## tolerance.
STREAMED_TOL = {"Newton": 1e-10, "L-BFGS": 1e-5}


@pytest.mark.parametrize("method", ["Newton", "L-BFGS"])
@pytest.mark.parametrize("batch_size", [7, 50, 100_000])
def test_FastMBAR_streamed_matches_unbatched(setup_mbar_data, method, batch_size):
    energy, num_conf, F_ref, energy_p, F_ref_p = setup_mbar_data
    tol = STREAMED_TOL[method]
    ref = FastMBAR(energy, num_conf, verbose=False)
    fastmbar = FastMBAR(
        energy,
        num_conf,
        verbose=False,
        method=method,
        batch_size=batch_size,
        energy_on_device=False,
    )
    assert fastmbar.energy_on_device is False
    assert fastmbar.F == approx(ref.F, abs=tol)
    assert fastmbar.F_cov == approx(ref.F_cov, abs=tol)
    assert fastmbar.log_prob_mix == approx(ref.log_prob_mix, abs=tol)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(energy_on_device=False), ValueError),
        (dict(batch_size=50, energy_on_device="no"), TypeError),
    ],
)
def test_FastMBAR_energy_on_device_errors(setup_mbar_data, kwargs, error):
    energy, num_conf, F_ref, energy_p, F_ref_p = setup_mbar_data
    with pytest.raises(error):
        FastMBAR(energy, num_conf, **kwargs)


@pytest.mark.parametrize("method", ["Newton", "L-BFGS"])
@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("batch_size", [7, 100_000])
def test_FastMBAR_streamed_perturbed_and_bootstrap_match_unbatched(
    setup_mbar_data, monkeypatch, method, bootstrap, batch_size
):
    energy, num_conf, F_ref, energy_p, F_ref_p = setup_mbar_data
    tol = STREAMED_TOL[method]
    ## the bootstrap is seeded from the clock; pin it so that both runs draw the
    ## same resamples
    monkeypatch.setattr("bayesmbar.fastmbar.time", lambda: 1234.5)

    kwargs = dict(bootstrap=bootstrap, bootstrap_num_rep=5, verbose=False)
    ref = FastMBAR(energy, num_conf, **kwargs)
    fastmbar = FastMBAR(
        energy,
        num_conf,
        method=method,
        batch_size=batch_size,
        energy_on_device=False,
        **kwargs,
    )
    assert fastmbar.F == approx(ref.F, abs=tol)
    assert fastmbar.F_std == approx(ref.F_std, abs=tol)
    assert fastmbar.log_prob_mix == approx(ref.log_prob_mix, abs=tol)

    results = fastmbar.calculate_free_energies_of_perturbed_states(energy_p)
    results_ref = ref.calculate_free_energies_of_perturbed_states(energy_p)
    assert results["F"] == approx(results_ref["F"], abs=tol)
    assert results["F_std"] == approx(results_ref["F_std"], abs=tol)


def test_FastMBAR_streamed_fortran_order_energy_is_not_copied(setup_mbar_data):
    energy, num_conf, F_ref, energy_p, F_ref_p = setup_mbar_data
    energy_f = np.asfortranarray(energy, dtype=np.float64)
    fastmbar = FastMBAR(energy_f, num_conf, batch_size=50, energy_on_device=False)
    assert np.shares_memory(fastmbar.energy, energy_f)
    assert fastmbar.F == approx(FastMBAR(energy, num_conf).F, abs=1e-10)


def test_FastMBAR_streamed_from_memmap(setup_mbar_data, tmp_path):
    ## an energy matrix in a file, opened Fortran-ordered, is streamed from the
    ## file without being copied into memory first
    energy, num_conf, F_ref, energy_p, F_ref_p = setup_mbar_data
    mm = np.memmap(tmp_path / "energy.dat", dtype=np.float64, mode="w+", shape=energy.shape, order="F")
    mm[:] = energy
    mm.flush()
    energy_mm = np.memmap(tmp_path / "energy.dat", dtype=np.float64, mode="r", shape=energy.shape, order="F")

    fastmbar = FastMBAR(energy_mm, num_conf, batch_size=50, energy_on_device=False)
    assert np.shares_memory(fastmbar.energy, energy_mm)
    assert fastmbar.F == approx(FastMBAR(energy, num_conf).F, abs=1e-10)
