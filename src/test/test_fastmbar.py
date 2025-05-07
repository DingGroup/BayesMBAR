import pytest
from pytest import approx
from bayesmbar import FastMBAR


@pytest.mark.parametrize("method", ["Newton", "L-BFGS-B"])
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