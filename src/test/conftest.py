import jax.random as jr
import jax.numpy as jnp
import numpy as np
import pytest

@pytest.fixture
def setup_data():
    num_states = 5
    key = jr.PRNGKey(0)
    key, subkey = jr.split(key)
    num_conf = jr.randint(subkey, (num_states,), 50, 100)

    key, subkey = jr.split(key)
    mu = jr.uniform(subkey, (num_states,), jnp.float64, 0, 2)

    key, subkey = jr.split(key)
    sigma = jr.uniform(subkey, (num_states,), jnp.float64, 1, 3)

    ## draw samples from each state and
    ## calculate energies of each sample in all states
    Xs = []
    for i in range(num_states):
        key, subkey = jr.split(key)
        Xs.append(jr.normal(subkey, (num_conf[i],)) * sigma[i] + mu[i])

    Xs = jnp.concatenate(Xs)
    Xs = Xs.reshape((-1, 1))
    energy = 0.5 * ((Xs - mu) / sigma) ** 2
    energy = energy.T

    F_ref = -jnp.log(sigma)
    pi = num_conf / num_conf.sum()
    F_ref = F_ref - jnp.sum(pi * F_ref)

    return np.array(energy), np.array(num_conf), np.array(F_ref), key