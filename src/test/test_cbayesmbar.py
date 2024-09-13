import pytest
from pytest import approx
from bayesmbar import CBayesMBAR
import numpy as np


@pytest.fixture
def setup_data_cbmbar():
    ka, kb, kc, kd = 4, 9, 16, 25
    ## equilibrium positions of three two-dimensional harmonic oscillators
    mu = {
        "a": np.array([-1.0, 1.0]),
        "b": np.array([1.0, 1.0]),
        "c": np.array([1.0, -1.0]),
        "d": np.array([-1.0, -1.0]),
    }

    ## spring constants
    k = {
        "a": np.ones(2) * ka,
        "b": np.ones(2) * kb,
        "c": np.ones(2) * kc,
        "d": np.ones(2) * kd,
    }

    ## interpolate the equilibrium positions and spring constants between pairs of states
    pairs = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c"), ("b", "d")]
    for s1, s2 in pairs:
        for idx in range(1, 4):
            mu[(s1, s2, idx)] = mu[s1] + (mu[s2] - mu[s1]) / 4 * idx
            k[(s1, s2, idx)] = k[s1] + (k[s2] - k[s1]) / 4 * idx

    n = 2000

    ## compute the free energy difference between pairs of states analytically
    def k_to_sigma(k):
        return np.sqrt(1.0 / k)

    deltaF_ref = {}
    for s1, s2 in pairs:
        deltaF_ref[(s1, s2)] = (
            -np.log(k_to_sigma(k[s2])).sum() + np.log(k_to_sigma(k[s1])).sum()
        )

    ## sample configurations from all states including both endpoints and intermediates
    np.random.seed(0)
    x = {}
    for s in mu.keys():
        x[s] = np.random.normal(mu[s], k_to_sigma(k[s]), (n, 2))

    u_list = []
    for s1, s2 in pairs:
        key = [s1] + [(s1, s2, idx) for idx in range(1, 4)] + [s2]
        xs = np.concatenate([x[s] for s in key])
        u = np.stack(
            [np.sum(0.5 * k[s] * (xs - mu[s]) ** 2, axis=1) for s in key], axis=0
        )
        u_list.append(u)

    num_conf_list = [
        [u.shape[1] // u.shape[0] for i in range(u.shape[0])] for u in u_list
    ]
    identical_states = (
        [(0, 0), (3, 4), (4, 0)],
        [(0, 4), (1, 0), (5, 0)],
        [(1, 4), (2, 0), (4, 4)],
        [(2, 4), (3, 0), (5, 4)],
    )

    return u_list, num_conf_list, identical_states, pairs, deltaF_ref


@pytest.mark.parametrize("method", ["Newton", "L-BFGS-B"])
def test_CBayesMBAR(setup_data_cbmbar, method):
    u_list, num_conf_list, identical_states, pairs, deltaF_ref = setup_data_cbmbar
    cbmbar = CBayesMBAR(
        u_list, num_conf_list, identical_states, method=method, random_seed=0
    )

    deltaF_cbmbar = {}
    for i in range(len(pairs)):
        s1, s2 = pairs[i]
        deltaF_cbmbar[(s1, s2)] = cbmbar.DeltaF_mode[i][0, -1].item()

    assert deltaF_cbmbar == approx(deltaF_ref, abs=1e-1)
