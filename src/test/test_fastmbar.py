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


def test_FastMBAR_batch_size_with_bootstrap_raises(setup_mbar_data):
    energy, num_conf, F_ref, energy_p, F_ref_p = setup_mbar_data
    with pytest.raises(ValueError):
        FastMBAR(energy, num_conf, bootstrap=True, batch_size=10)


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
