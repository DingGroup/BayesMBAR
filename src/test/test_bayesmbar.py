import numpy as np
import pytest
from pytest import approx
from bayesmbar import BayesMBAR


@pytest.mark.parametrize("method", ["Newton", "L-BFGS-B"])
def test_BayesMBAR(setup_mbar_data, method):
    energy, num_conf, F_ref, energy_p, F_ref_p = setup_mbar_data
    mbar = BayesMBAR(
        energy,
        num_conf,
        verbose=True,
        method=method,
    )
    assert mbar.F_mode == approx(F_ref, abs=1e-1)
    assert mbar.F_mean == approx(F_ref, abs=1e-1)


def test_two_states():
    M = 2  ## number of states
    mu = np.linspace(0, 1, M)  ## equilibrium positions
    k = np.random.uniform(10, 30, M)  ## force constants
    sigma = np.sqrt(1.0 / k)
    F_reference = -np.log(sigma)
    F_reference -= F_reference[0]
    n = 100
    x = [np.random.normal(mu[i], sigma[i], (n,)) for i in range(M)]
    x = np.concatenate(x)
    u = 0.5 * k.reshape((-1, 1)) * (x - mu.reshape((-1, 1))) ** 2
    num_conf = np.array([n for i in range(M)])
    mbar = BayesMBAR(
        u,
        num_conf,
        prior="uniform",
        mean=None,
        state_cv=None,
        kernel=None,
        sample_size=1000,
        warmup_steps=100,
        optimize_steps=0,
        random_seed=0,
        verbose=False,
    )
    F_reference = F_reference[-1] - F_reference[0]
    F_mean = mbar.F_mean
    F_mode = mbar.F_mode
    assert (F_mean[-1] - F_mean[0]) == approx(F_reference, abs=1e-6)
    assert (F_mode[-1] - F_mode[0]) == approx(F_reference, abs=1e-6)

    # results = fastmbar.calculate_free_energies_of_perturbed_states(energy_p)
    # results['F'] = results['F'] - results['F'].mean()
    # assert results['F'] == approx(F_ref_p, abs = 1e-1)


# import pytest
# from pytest import approx
# import jax.numpy as jnp
# from jax import random
# from bayesmbar import BayesMBAR


# # @pytest.fixture
# def setup_data():
#     num_states = 5
#     key = random.PRNGKey(0)
#     key, subkey = random.split(key)
#     num_conf = random.randint(subkey, (num_states,), 50, 100)

#     key, subkey = random.split(key)
#     mu = random.uniform(subkey, (num_states,), jnp.float64, 0, 2)

#     key, subkey = random.split(key)
#     sigma = random.uniform(subkey, (num_states,), jnp.float64, 1, 3)

#     ## draw samples from each state and
#     ## calculate energies of each sample in all states
#     Xs = []
#     for i in range(num_states):
#         key, subkey = random.split(key)
#         Xs.append(random.normal(subkey, (num_conf[i],)) * sigma[i] + mu[i])

#     Xs = jnp.concatenate(Xs)
#     Xs = Xs.reshape((-1, 1))
#     energy = 0.5 * ((Xs - mu) / sigma) ** 2
#     energy = energy.T

#     F_ref = -jnp.log(sigma)
#     pi = num_conf / num_conf.sum()
#     F_ref = F_ref - jnp.sum(pi * F_ref)

#     return energy, num_conf, F_ref, key


# energy, num_conf, F_ref, key = setup_data()

# num_states = len(num_conf)
# key, subkey = random.split(key)
# cv = random.normal(subkey, (num_states, 1))
# d2 = bayesmbar._compute_squared_distance(cv)
# params = {'scale': 1.0, 'length_scale': 1.0}

# key, subkey = random.split(key)
# mbar = BayesMBAR(
#     energy, num_conf, subkey, 'normal', cv, 'SE', sample_size=2000, warmup_steps=200
# )

# @pytest.mark.parametrize("method", ["Newton", "L-BFGS-B"])
# @pytest.mark.parametrize("bootstrap", [False, True])
# def test_BayesMBAR(setup_data, method, bootstrap):
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
