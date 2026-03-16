import pytest
from pytest import approx
import numpy as np
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

    # results = fastmbar.calculate_free_energies_of_perturbed_states(energy_p)
    # results['F'] = results['F'] - results['F'].mean()
    # assert results['F'] == approx(F_ref_p, abs = 1e-1)


class TestAdaptiveParameters:
    """Tests for adaptive sample_size, warmup_steps, and early stopping parameters."""

    def test_adaptive_sample_size_small_problem(self, setup_mbar_data):
        """Test that sample_size defaults to 1000 for small problems (m-1 < 10)."""
        energy, num_conf, F_ref, _, _ = setup_mbar_data
        m = energy.shape[0]  # 5 states
        mbar = BayesMBAR(energy, num_conf, verbose=False)

        # For 5 states: max(1000, 100 * 4) = 1000
        expected_sample_size = max(1000, 100 * (m - 1))
        assert mbar._sample_size == expected_sample_size
        assert mbar._sample_size == 1000

    def test_adaptive_sample_size_large_problem(self):
        """Test that sample_size scales with dimension for larger problems."""
        # Create a larger problem with 15 states
        m = 15
        n = 100
        np.random.seed(42)
        energy = np.random.randn(m, n)
        num_conf = np.array([n // m] * m)

        mbar = BayesMBAR(energy, num_conf, verbose=False)

        # For 15 states: max(1000, 100 * 14) = 1400
        expected_sample_size = max(1000, 100 * (m - 1))
        assert mbar._sample_size == expected_sample_size
        assert mbar._sample_size == 1400

    def test_adaptive_warmup_steps_default(self, setup_mbar_data):
        """Test that warmup_steps defaults to max(500, sample_size // 2)."""
        energy, num_conf, _, _, _ = setup_mbar_data
        mbar = BayesMBAR(energy, num_conf, verbose=False)

        # sample_size=1000, so warmup = max(500, 500) = 500
        expected_warmup = max(500, mbar._sample_size // 2)
        assert mbar._warmup_steps == expected_warmup

    def test_adaptive_warmup_steps_scales_with_sample_size(self):
        """Test that warmup_steps scales when sample_size is large."""
        m = 15
        n = 100
        np.random.seed(42)
        energy = np.random.randn(m, n)
        num_conf = np.array([n // m] * m)

        mbar = BayesMBAR(energy, num_conf, verbose=False)

        # sample_size=1400, so warmup = max(500, 700) = 700
        expected_warmup = max(500, mbar._sample_size // 2)
        assert mbar._warmup_steps == expected_warmup
        assert mbar._warmup_steps == 700

    def test_explicit_sample_size_overrides_adaptive(self, setup_mbar_data):
        """Test that explicit sample_size overrides the adaptive default."""
        energy, num_conf, _, _, _ = setup_mbar_data
        explicit_sample_size = 2000

        mbar = BayesMBAR(energy, num_conf, sample_size=explicit_sample_size, verbose=False)

        assert mbar._sample_size == explicit_sample_size

    def test_explicit_warmup_steps_overrides_adaptive(self, setup_mbar_data):
        """Test that explicit warmup_steps overrides the adaptive default."""
        energy, num_conf, _, _, _ = setup_mbar_data
        explicit_warmup = 300

        mbar = BayesMBAR(energy, num_conf, warmup_steps=explicit_warmup, verbose=False)

        assert mbar._warmup_steps == explicit_warmup

    def test_early_stopping_parameters_defaults(self, setup_mbar_data):
        """Test that early stopping parameters have correct defaults."""
        energy, num_conf, _, _, _ = setup_mbar_data
        mbar = BayesMBAR(energy, num_conf, verbose=False)

        assert mbar._max_optimize_steps == 10000
        assert mbar._optimize_patience == 500
        assert mbar._optimize_min_delta == 1e-4

    def test_early_stopping_parameters_custom(self, setup_mbar_data):
        """Test that custom early stopping parameters are set correctly."""
        energy, num_conf, _, _, _ = setup_mbar_data
        mbar = BayesMBAR(
            energy,
            num_conf,
            max_optimize_steps=5000,
            optimize_patience=200,
            optimize_min_delta=1e-5,
            verbose=False,
        )

        assert mbar._max_optimize_steps == 5000
        assert mbar._optimize_patience == 200
        assert mbar._optimize_min_delta == 1e-5

    def test_combined_explicit_and_adaptive(self, setup_mbar_data):
        """Test mixing explicit and adaptive parameters."""
        energy, num_conf, _, _, _ = setup_mbar_data

        # Set explicit sample_size, let warmup_steps be adaptive
        mbar = BayesMBAR(energy, num_conf, sample_size=1500, verbose=False)

        assert mbar._sample_size == 1500
        # warmup should adapt to the explicit sample_size
        assert mbar._warmup_steps == max(500, 1500 // 2)
        assert mbar._warmup_steps == 750

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
