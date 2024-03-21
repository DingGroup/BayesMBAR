import pytest
from pytest import approx
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
import jax.scipy.stats as stats
import math
import os, sys
from bayesbar import BayesBAR
import os

# jax.config.update("jax_platform_name", "cpu")

# @pytest.fixture
def setup_data():
    num_states = 2
    key = random.PRNGKey(10)
    key, subkey = random.split(key)
    num_conf = random.randint(subkey, (num_states,), 50, 100)

    key, subkey = random.split(key)
    mu = random.uniform(subkey, (num_states,), jnp.float64, 0, 2)
    

    key, subkey = random.split(key)
    sigma = random.uniform(subkey, (num_states,), jnp.float64, 1, 3)

    num_conf = jnp.array([100, 100])
    mu = jnp.array([0, 60])
    sigma = jnp.array([1, 3])
    

    ## draw samples from each state and
    ## calculate energies of each sample in all states
    Xs = []
    for i in range(num_states):
        key, subkey = random.split(key)
        Xs.append(random.normal(subkey, (num_conf[i],)) * sigma[i] + mu[i])

    Xs = jnp.concatenate(Xs)
    Xs = Xs.reshape((-1, 1))
    energy = 0.5 * ((Xs - mu) / sigma) ** 2
    energy = energy.T

    F_ref = -jnp.log(sigma)
    pi = num_conf / num_conf.sum()
    F_ref = F_ref - jnp.sum(pi * F_ref)

    return energy, num_conf, F_ref, key


energy, num_conf, F_ref, key = setup_data()
key, subkey = random.split(key)
bar = BayesBAR(energy, num_conf, verbose=True, sample_size=0)
