import pytest
from pytest import approx
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from bayesmbar import BayesMBAR


@pytest.fixture
def small_synthetic_data():
    """Create a small synthetic dataset for testing lazy evaluation."""
    num_states = 3
    key = jr.PRNGKey(42)
    key, subkey = jr.split(key)
    num_conf = jr.randint(subkey, (num_states,), 30, 50)

    key, subkey = jr.split(key)
    mu = jr.uniform(subkey, (num_states,), jnp.float64, 0, 2)

    key, subkey = jr.split(key)
    sigma = jr.uniform(subkey, (num_states,), jnp.float64, 1, 3)

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

    # Create state_cv for normal prior tests (1D collective variable)
    state_cv = jnp.linspace(0, 1, num_states).reshape(-1, 1)

    return np.array(energy), np.array(num_conf), np.array(F_ref), np.array(state_cv)


@pytest.fixture
def two_state_data():
    """Create a minimal 2-state dataset for edge case testing."""
    num_states = 2
    key = jr.PRNGKey(123)
    key, subkey = jr.split(key)
    num_conf = jr.randint(subkey, (num_states,), 40, 60)

    key, subkey = jr.split(key)
    mu = jr.uniform(subkey, (num_states,), jnp.float64, 0, 2)

    key, subkey = jr.split(key)
    sigma = jr.uniform(subkey, (num_states,), jnp.float64, 1, 3)

    Xs = []
    for i in range(num_states):
        key, subkey = jr.split(key)
        Xs.append(jr.normal(subkey, (num_conf[i],)) * sigma[i] + mu[i])

    Xs = jnp.concatenate(Xs)
    Xs = Xs.reshape((-1, 1))
    energy = 0.5 * ((Xs - mu) / sigma) ** 2
    energy = energy.T

    state_cv = jnp.linspace(0, 1, num_states).reshape(-1, 1)

    return np.array(energy), np.array(num_conf), np.array(state_cv)


@pytest.mark.parametrize("method", ["Newton", "L-BFGS-B"])
def test_BayesMBAR(setup_mbar_data, method):
    energy, num_conf, F_ref, energy_p, F_ref_p = setup_mbar_data
    mbar = BayesMBAR(
        energy,
        num_conf,
        verbose=True,
        method=method,
    )
    assert mbar.F_mode == approx(F_ref, abs = 1e-1)
    assert mbar.F_mean == approx(F_ref, abs = 1e-1)

    #results = fastmbar.calculate_free_energies_of_perturbed_states(energy_p)
    #results['F'] = results['F'] - results['F'].mean()
    #assert results['F'] == approx(F_ref_p, abs = 1e-1)


class TestBayesMBARLazyEvaluation:
    """Tests for BayesMBAR lazy evaluation and property access."""

    def test_no_computation_at_construction_uniform_prior(self, small_synthetic_data):
        """Verify no computation is triggered at construction with uniform prior."""
        energy, num_conf, F_ref, state_cv = small_synthetic_data

        mbar = BayesMBAR(
            energy,
            num_conf,
            prior="uniform",
            verbose=False,
        )

        # Check that lazy evaluation flags are all False
        assert mbar._mode_ll_computed is False
        assert mbar._samples_ll_computed is False
        assert mbar._posterior_mode_computed is False
        assert mbar._posterior_samples_computed is False

        # Verify internal cached fields do not exist yet
        assert not hasattr(mbar, "_dF_mode_ll") or not hasattr(mbar, "_F_mode_ll")
        assert not hasattr(mbar, "_dF_samples_ll") or not hasattr(mbar, "_F_samples_ll")

    def test_F_mode_triggers_mode_computation_uniform(self, small_synthetic_data):
        """Access F_mode and verify it triggers likelihood-mode computation."""
        energy, num_conf, F_ref, state_cv = small_synthetic_data
        m = energy.shape[0]

        mbar = BayesMBAR(
            energy,
            num_conf,
            prior="uniform",
            verbose=False,
        )

        # Access F_mode
        F_mode = mbar.F_mode

        # Verify shape is (m,)
        assert F_mode.shape == (m,)

        # Verify values are finite
        assert np.all(np.isfinite(F_mode))

        # Verify mode computation was triggered
        assert mbar._mode_ll_computed is True

        # Verify samples computation was NOT triggered (lazy)
        assert mbar._samples_ll_computed is False

    def test_F_mean_F_cov_F_samples_shapes_uniform(self, small_synthetic_data):
        """Access F_mean, F_cov, F_samples and verify shapes and finite values."""
        energy, num_conf, F_ref, state_cv = small_synthetic_data
        m = energy.shape[0]
        sample_size = 100

        mbar = BayesMBAR(
            energy,
            num_conf,
            prior="uniform",
            sample_size=sample_size,
            warmup_steps=50,
            verbose=False,
        )

        # Access F_mean
        F_mean = mbar.F_mean
        assert F_mean.shape == (m,), f"Expected F_mean shape ({m},), got {F_mean.shape}"
        assert np.all(np.isfinite(F_mean)), "F_mean contains non-finite values"

        # Access F_cov
        F_cov = mbar.F_cov
        assert F_cov.shape == (m, m), f"Expected F_cov shape ({m}, {m}), got {F_cov.shape}"
        assert np.all(np.isfinite(F_cov)), "F_cov contains non-finite values"

        # Access F_samples
        F_samples = mbar.F_samples
        assert F_samples.shape == (sample_size, m), f"Expected F_samples shape ({sample_size}, {m}), got {F_samples.shape}"
        assert np.all(np.isfinite(F_samples)), "F_samples contains non-finite values"

        # Verify samples computation was triggered
        assert mbar._samples_ll_computed is True

    def test_F_mode_F_mean_shapes_normal_prior(self, small_synthetic_data):
        """For prior='normal', verify F_mode and F_mean shapes and posterior path runs."""
        energy, num_conf, F_ref, state_cv = small_synthetic_data
        m = energy.shape[0]

        mbar = BayesMBAR(
            energy,
            num_conf,
            prior="normal",
            state_cv=state_cv,
            sample_size=50,
            warmup_steps=20,
            optimize_steps=100,  # Small for faster test
            verbose=False,
        )

        # Verify no computation at construction
        assert mbar._mode_ll_computed is False
        assert mbar._posterior_mode_computed is False
        assert mbar._posterior_samples_computed is False

        # Access F_mode (triggers hyperparameter optimization + posterior mode)
        F_mode = mbar.F_mode
        assert F_mode.shape == (m,), f"Expected F_mode shape ({m},), got {F_mode.shape}"
        assert np.all(np.isfinite(F_mode)), "F_mode contains non-finite values"

        # Verify posterior mode was computed
        assert mbar._posterior_mode_computed is True

        # Access F_mean (triggers posterior sampling)
        F_mean = mbar.F_mean
        assert F_mean.shape == (m,), f"Expected F_mean shape ({m},), got {F_mean.shape}"
        assert np.all(np.isfinite(F_mean)), "F_mean contains non-finite values"

        # Verify posterior samples were computed
        assert mbar._posterior_samples_computed is True

    def test_idempotent_property_access_uniform(self, small_synthetic_data):
        """Verify repeated property access returns same values and does not recompute."""
        energy, num_conf, F_ref, state_cv = small_synthetic_data

        mbar = BayesMBAR(
            energy,
            num_conf,
            prior="uniform",
            sample_size=100,
            warmup_steps=50,
            verbose=False,
        )

        # First access
        F_mode_1 = mbar.F_mode
        F_mean_1 = mbar.F_mean
        F_cov_1 = mbar.F_cov
        F_samples_1 = mbar.F_samples

        # Store cached internal state
        dF_mode_ll_cached = mbar._dF_mode_ll.copy()
        dF_samples_ll_cached = mbar._dF_samples_ll.copy()

        # Second access
        F_mode_2 = mbar.F_mode
        F_mean_2 = mbar.F_mean
        F_cov_2 = mbar.F_cov
        F_samples_2 = mbar.F_samples

        # Verify outputs are identical (idempotent)
        np.testing.assert_array_equal(F_mode_1, F_mode_2, "F_mode changed on repeated access")
        np.testing.assert_array_equal(F_mean_1, F_mean_2, "F_mean changed on repeated access")
        np.testing.assert_array_equal(F_cov_1, F_cov_2, "F_cov changed on repeated access")
        np.testing.assert_array_equal(F_samples_1, F_samples_2, "F_samples changed on repeated access")

        # Verify internal cached fields were not recomputed (same objects)
        np.testing.assert_array_equal(
            mbar._dF_mode_ll, dF_mode_ll_cached,
            "Internal _dF_mode_ll changed, suggesting recomputation"
        )
        np.testing.assert_array_equal(
            mbar._dF_samples_ll, dF_samples_ll_cached,
            "Internal _dF_samples_ll changed, suggesting recomputation"
        )

    def test_idempotent_property_access_normal(self, small_synthetic_data):
        """Verify repeated property access is idempotent for normal prior."""
        energy, num_conf, F_ref, state_cv = small_synthetic_data

        mbar = BayesMBAR(
            energy,
            num_conf,
            prior="normal",
            state_cv=state_cv,
            sample_size=50,
            warmup_steps=20,
            optimize_steps=100,
            verbose=False,
        )

        # First access
        F_mode_1 = mbar.F_mode
        F_mean_1 = mbar.F_mean

        # Second access
        F_mode_2 = mbar.F_mode
        F_mean_2 = mbar.F_mean

        # Verify outputs are identical
        np.testing.assert_array_equal(F_mode_1, F_mode_2, "F_mode changed on repeated access (normal prior)")
        np.testing.assert_array_equal(F_mean_1, F_mean_2, "F_mean changed on repeated access (normal prior)")

        # Verify computation flags remain True (no re-trigger)
        assert mbar._posterior_mode_computed is True
        assert mbar._posterior_samples_computed is True


class TestBayesMBARCovarianceEdgeCases:
    """Tests for covariance matrix edge cases and post-processing."""

    def test_covariance_2d_for_two_states(self, two_state_data):
        """Validate covariance matrices are at least 2D even for m=2."""
        energy, num_conf, state_cv = two_state_data
        m = energy.shape[0]
        assert m == 2, "This test requires exactly 2 states"

        mbar = BayesMBAR(
            energy,
            num_conf,
            prior="uniform",
            sample_size=100,
            warmup_steps=50,
            verbose=False,
        )

        F_cov = mbar.F_cov

        # Verify covariance is 2D
        assert F_cov.ndim == 2, f"Expected 2D covariance, got {F_cov.ndim}D"
        assert F_cov.shape == (m, m), f"Expected shape ({m}, {m}), got {F_cov.shape}"

    def test_covariance_no_negative_diagonals_after_postprocessing(self, small_synthetic_data):
        """Validate covariance matrix diagonals are non-negative after post-processing."""
        energy, num_conf, F_ref, state_cv = small_synthetic_data

        mbar = BayesMBAR(
            energy,
            num_conf,
            prior="uniform",
            sample_size=100,
            warmup_steps=50,
            verbose=False,
        )

        F_cov = mbar.F_cov
        diagonals = np.diag(F_cov)

        # Verify all diagonal elements are non-negative
        assert np.all(diagonals >= 0), f"Negative diagonal elements found: {diagonals[diagonals < 0]}"

        # Verify finite values
        assert np.all(np.isfinite(diagonals)), "Diagonal contains non-finite values"

    def test_covariance_no_negative_diagonals_two_states(self, two_state_data):
        """Validate no negative diagonals for m=2 edge case after post-processing."""
        energy, num_conf, state_cv = two_state_data

        mbar = BayesMBAR(
            energy,
            num_conf,
            prior="uniform",
            sample_size=100,
            warmup_steps=50,
            verbose=False,
        )

        F_cov = mbar.F_cov
        diagonals = np.diag(F_cov)

        # Verify all diagonal elements are non-negative
        assert np.all(diagonals >= 0), f"Negative diagonal elements in 2-state case: {diagonals}"

        # Verify the covariance matrix is symmetric
        np.testing.assert_array_almost_equal(F_cov, F_cov.T, decimal=10, err_msg="Covariance matrix is not symmetric")

    def test_F_std_non_negative(self, small_synthetic_data):
        """Verify F_std returns non-negative standard deviations."""
        energy, num_conf, F_ref, state_cv = small_synthetic_data
        m = energy.shape[0]

        mbar = BayesMBAR(
            energy,
            num_conf,
            prior="uniform",
            sample_size=100,
            warmup_steps=50,
            verbose=False,
        )

        F_std = mbar.F_std

        # Verify shape
        assert F_std.shape == (m,), f"Expected F_std shape ({m},), got {F_std.shape}"

        # Verify all standard deviations are non-negative
        assert np.all(F_std >= 0), f"Negative standard deviation found: {F_std[F_std < 0]}"

        # Verify finite values
        assert np.all(np.isfinite(F_std)), "F_std contains non-finite values"


class TestBayesMBARDeltaF:
    """Tests for DeltaF properties."""

    def test_DeltaF_mode_shape_and_symmetry(self, small_synthetic_data):
        """Verify DeltaF_mode has correct shape and antisymmetric structure."""
        energy, num_conf, F_ref, state_cv = small_synthetic_data
        m = energy.shape[0]

        mbar = BayesMBAR(
            energy,
            num_conf,
            prior="uniform",
            verbose=False,
        )

        DeltaF_mode = mbar.DeltaF_mode

        # Verify shape
        assert DeltaF_mode.shape == (m, m), f"Expected shape ({m}, {m}), got {DeltaF_mode.shape}"

        # Verify antisymmetric: DeltaF[i,j] = -DeltaF[j,i]
        np.testing.assert_array_almost_equal(
            DeltaF_mode, -DeltaF_mode.T, decimal=10,
            err_msg="DeltaF_mode is not antisymmetric"
        )

        # Verify diagonal is zero
        np.testing.assert_array_almost_equal(
            np.diag(DeltaF_mode), np.zeros(m), decimal=10,
            err_msg="DeltaF_mode diagonal is not zero"
        )

    def test_DeltaF_std_shape_and_symmetry(self, small_synthetic_data):
        """Verify DeltaF_std has correct shape and symmetric structure."""
        energy, num_conf, F_ref, state_cv = small_synthetic_data
        m = energy.shape[0]

        mbar = BayesMBAR(
            energy,
            num_conf,
            prior="uniform",
            sample_size=200,
            warmup_steps=100,
            verbose=False,
            random_seed=500,  # Fixed seed for reproducibility
        )

        DeltaF_std = mbar.DeltaF_std

        # Verify shape
        assert DeltaF_std.shape == (m, m), f"Expected shape ({m}, {m}), got {DeltaF_std.shape}"

        # Verify symmetric: std[i,j] = std[j,i]
        np.testing.assert_array_almost_equal(
            DeltaF_std, DeltaF_std.T, decimal=10,
            err_msg="DeltaF_std is not symmetric"
        )

        # Verify diagonal is zero (no uncertainty in F_i - F_i)
        np.testing.assert_array_almost_equal(
            np.diag(DeltaF_std), np.zeros(m), decimal=10,
            err_msg="DeltaF_std diagonal is not zero"
        )

        # Verify all values are finite and non-negative
        assert np.all(np.isfinite(DeltaF_std)), "DeltaF_std contains non-finite values"
        assert np.all(DeltaF_std >= 0), "DeltaF_std contains negative values"

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
