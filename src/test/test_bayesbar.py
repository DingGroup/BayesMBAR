import pytest
from pytest import approx
from bayesmbar import BayesBAR

@pytest.mark.parametrize("method", ["Newton", "L-BFGS-B"])
def test_BayesBAR(setup_bar_data, method):
    energy, num_conf, DeltaF_ref = setup_bar_data
    energy = energy[0:2]
    num_conf = num_conf[0:2]
    bar = BayesBAR(
        energy,
        num_conf,
        method=method,        
        verbose=False,        
    )
    assert bar.DeltaF_mode.item() == approx(DeltaF_ref, abs = 1e-1)
    assert bar.DeltaF_mean.item() == approx(DeltaF_ref, abs = 1e-1)
