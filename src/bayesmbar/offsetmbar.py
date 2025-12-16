import time
from typing import List
import numpy as np
from numpy import ndarray
import jax
import jax.numpy as jnp
from jax import random, hessian, jit, value_and_grad, vmap
from scipy import optimize

from .bayesmbar import _sample_from_logdensity
from .utils import fmin_newton, _compute_log_likelihood_of_dF

jax.config.update("jax_enable_x64", True)


class OffsetMBAR:
    def __init__(
        self,
        energies: list[np.ndarray],
        nums_conf: list[np.ndarray],
        offsets: list[float],
        sample_size: int = 1000,
        warmup_steps: int = 500,
        method: str = "Newton",
        max_iter: int = 300,
        random_seed: int = None,
        verbose: bool = True,
    ) -> None:
        """
        Offset-constrained Coupled BayesMBAR.

        This variant of Coupled BayesMBAR enforces a simple constraint: for each
        coupled system i, the free-energy difference computed from its energies
        plus a provided scalar offset must be identical across all systems. In
        other words, the solution to the MBAR equation satisfies
        mbar(energies[i]) + offsets[i] = constant for every system i.

        Parameters
        ----------
        energies : List[numpy.ndarray]
            Per-system arrays of reduced potentials (in units of kT) used by
            MBAR. One array per coupled system.
        nums_conf : List[numpy.ndarray]
            Per-system arrays of counts (number of configurations) corresponding
            to the provided energies.
        offsets : List[float]
            One scalar offset per system. The constraint is that
            mbar(energies[i]) + offsets[i] is the same across all systems.
        sample_size : int, optional
            Number of samples to draw from the likelihood. Default: 1000.
        warmup_steps : int, optional
            Number of warmup steps for the HMC sampler. Default: 500.
        method : str, optional
            Optimization method to find the likelihood mode. Either "Newton" or
            "L-BFGS-B". Default: "Newton".
        max_iter : int, optional
            Maximum number of iterations for the Newton optimizer. Default: 300.
        random_seed : int, optional
            Random seed. If None, a seed is generated from the current time.
            Default: None.
        verbose : bool, optional
            If True, print sampling progress. Default: True.
        """
        for i, (energy, num_conf) in enumerate(zip(energies, nums_conf, strict=True)):
            n_lambda, n_conf = energy.shape
            assert n_lambda == len(num_conf), f"System {i}: energy has {n_lambda} lambda states, but num_conf has {len(num_conf)}."
            assert n_conf == sum(num_conf), f"System {i}: energy has {n_conf} configurations, but num_conf sums to {sum(num_conf)}."
        assert len(offsets) == len(energies), f"offsets length ({len(offsets)}) != energies length ({len(energies)})."

        self._energies = [jnp.float64(u) for u in energies]
        self._nums_conf = [jnp.int32(n) for n in nums_conf]
        self._offsets = jnp.array(offsets)
        
        self._sample_size = sample_size
        self._warmup_steps = warmup_steps
        self._verbose = verbose
        
        if random_seed is None:
            random_seed = int(time.time())
        self._rng_key = jax.random.PRNGKey(random_seed)
        
        # Number of states in each MBAR system
        self._nums_state = [len(n) for n in self._nums_conf]
        
        # Build the constraint matrix and offset vector
        # For each system, we have:
        # - (M-1) free energy differences for the MBAR states
        # - 1 additional offset parameter
        # Constraints:
        # 1. The offset state for all systems should be equal
        # 2. The difference between offset state and last MBAR state equals the offset value
        
        self._Q, self._b = _compute_offset_projection(self._nums_state, self._offsets)
        
        # Initialize x (the free parameters)
        x = jnp.zeros((self._Q.shape[1]))
        
        if self._verbose:
            print("Solve for the mode of the likelihood")
        
        if method == "Newton":
            f = jit(value_and_grad(_compute_offset_loss_likelihood))
            hess = jit(hessian(_compute_offset_loss_likelihood))
            res = fmin_newton(
                f,
                hess,
                x,
                args=(self._Q, self._b, self._energies, self._nums_conf),
                verbose=self._verbose,
                max_iter=max_iter,
            )
        elif method == "L-BFGS-B":
            options = {"disp": verbose, "gtol": 1e-8, "maxiter": max_iter}
            f = jit(value_and_grad(_compute_offset_loss_likelihood))
            res = optimize.minimize(
                lambda x: [
                    np.array(r) for r in f(x, self._Q, self._b, self._energies, self._nums_conf)
                ],
                x,
                jac=True,
                method="L-BFGS-B",
                tol=1e-12,
                options=options,
            )
            if not res.success:
                raise RuntimeError(
                    f"L-BFGS-B did not converge after {max_iter} iterations. "
                    f"Message: {res.message}. Consider increasing max_iter."
                )
        else:
            raise ValueError("Invalid method")
        
        x_mode_ll = res["x"]
        self._dF_mode_ll = jnp.dot(self._Q, x_mode_ll) + self._b
        self._state_F_mode_ll = _dF_to_state_F_with_offset(
            self._dF_mode_ll, self._nums_state
        )
        
        if self._sample_size > 0:
            if self._verbose:
                print("=====================================================")
                print("Sample from the likelihood")
            
            self._rng_key, subkey = random.split(self._rng_key)
            
            def logdensity(x):
                return _compute_offset_log_likelihood(
                    x, self._Q, self._b, self._energies, self._nums_conf
                )
            
            self._x_samples_ll = _sample_from_logdensity(
                subkey,
                x_mode_ll,
                logdensity,
                self._warmup_steps,
                self._sample_size,
                self._verbose,
            )
            self._dF_samples_ll = self._x_samples_ll @ self._Q.T + self._b
            
            self._state_F_samples_ll = vmap(_dF_to_state_F_with_offset, in_axes=[0, None])(
                self._dF_samples_ll, self._nums_state
            )
            self._state_F_mean_ll = [
                jnp.mean(F, axis=0) for F in self._state_F_samples_ll
            ]

    @property
    def F_mode(self) -> List[ndarray]:
        """The mode of free energies of all states in all MBAR systems.
        The free energy of state 0 in each system is set to 0.
        Each system has an additional last state representing the offset constraint.
        """
        return [
            np.array(jax.device_put(F, jax.devices("cpu")[0]))
            for F in self._state_F_mode_ll
        ]

    @property
    def F_samples(self) -> List[ndarray]:
        """The samples of free energies of all states in all MBAR systems.
        The free energy of state 0 in each system is set to 0.
        Each system has an additional last state representing the offset constraint.
        """
        return [
            np.array(jax.device_put(F, jax.devices("cpu")[0]))
            for F in self._state_F_samples_ll
        ]

    @property
    def F_mean(self) -> List[ndarray]:
        """The mean of free energies of all states in all MBAR systems.
        The free energy of state 0 in each system is set to 0.
        Each system has an additional last state representing the offset constraint.
        """
        return [
            np.array(jax.device_put(F, jax.devices("cpu")[0]))
            for F in self._state_F_mean_ll
        ]

    @property
    def DeltaF_mode(self) -> List[ndarray]:
        """The mode of free energy differences between all pairs of states in every MBAR system."""
        return [F[None, :] - F[:, None] for F in self.F_mode]

    @property
    def DeltaF_mean(self) -> List[ndarray]:
        """The mean of free energy differences between all pairs of states in every MBAR system."""
        return [F[None, :] - F[:, None] for F in self.F_mean]

    @property
    def DeltaF_std(self) -> List[ndarray]:
        """The standard deviation of free energy differences between all pairs of states."""
        return [np.std(F[:, None, :] - F[:, :, None], 0) for F in self.F_samples]


def _dF_to_state_F_with_offset(
    dF: jnp.ndarray, nums_state: List[int]
) -> List[jnp.ndarray]:
    """Convert the vector of free energy differences to state free energies.
    
    For each system i:
    - Extract the free energy differences for the MBAR states (M-1 values)
    - Add the offset state with value from dF
    
    The dF vector has structure (after Q @ x + b):
    [system0_dF_0, ..., system0_dF_{M-2}, system0_offset,
     system1_dF_0, ..., system1_dF_{M-2}, system1_offset,
     ...]
    where dF_k = F[k+1] - F[0] and the offset values already incorporate the constraints.
    """
    state_F = []
    idx = 0
    for i, n in enumerate(nums_state):
        # Extract MBAR free energy differences (n-1 values)
        mbar_dF = dF[..., idx : idx + n - 1]
        # Extract the offset free energy difference (1 value)
        offset_value = dF[..., idx + n - 1]
        
        # Build the full free energy vector: [0, dF_0, dF_1, ..., dF_{M-2}, offset_value]
        if mbar_dF.ndim == 0:
            # Single value case (M=2, only one MBAR state)
            F = jnp.array([0.0, mbar_dF, offset_value])
        elif mbar_dF.ndim == 1:
            # Vector case
            F = jnp.concatenate([jnp.zeros(1), mbar_dF, jnp.array([offset_value])])
        else:
            # Batch case (for sampling)
            F = jnp.concatenate([
                jnp.zeros((mbar_dF.shape[0], 1)),
                mbar_dF,
                offset_value[..., None]
            ], axis=-1)
        
        state_F.append(F)
        idx += n  # Move to next system (n-1 MBAR states + 1 offset state)
    
    return state_F


def _compute_offset_log_likelihood(
    x: jnp.ndarray,
    Q: jnp.ndarray,
    b: jnp.ndarray,
    energies: List[jnp.ndarray],
    nums_conf: List[jnp.ndarray],
):
    """Compute the log likelihood for the offset-constrained MBAR."""
    dF_full = jnp.dot(Q, x) + b
    nums_state = [len(s) for s in nums_conf]
    
    log_likelihood = 0
    idx = 0
    for i, n in enumerate(nums_state):
        # Extract the MBAR free energy differences (excluding the offset state)
        mbar_dF = dF_full[idx : idx + n - 1]
        
        # Compute log likelihood for this MBAR system
        log_likelihood += _compute_log_likelihood_of_dF(mbar_dF, energies[i], nums_conf[i])
        
        idx += n  # Move to next system
    
    return log_likelihood


def _compute_offset_loss_likelihood(
    x: jnp.ndarray,
    Q: jnp.ndarray,
    b: jnp.ndarray,
    energies: List[jnp.ndarray],
    nums_conf: List[jnp.ndarray],
):
    """Compute the loss (negative normalized log likelihood) for optimization."""
    log_likelihood = _compute_offset_log_likelihood(x, Q, b, energies, nums_conf)
    N = jnp.sum(jnp.array([jnp.sum(n) for n in nums_conf]))
    return -log_likelihood / N


def _compute_offset_projection(nums_state: List[int], offsets: jnp.ndarray):
    """Compute the projection matrix Q and offset vector b for the offset constraint.
    
    Returns Q and b such that: dF = Q @ x + b
    
    Constraint: F[i][-2] + offset[i] = constant for all i
    where F[i][-2] is the last MBAR state and F[i][-1] is the added offset state.
    
    Since dF[k] = F[k+1], we have F[i][-2] = F[i][M-1] = dF[i][M-2]
    
    So: dF[i][M-2] + offset[i] = constant (call it c)
    Or: dF[i][M-2] = c - offset[i]
    
    Parameterization:
    - For system 0: dF[0][0], ..., dF[0][M-3] are independent
    - For system i>0: dF[i][0], ..., dF[i][M-3] are independent
    - x[-1] represents the shared constant c
    
    Then:
    - dF[i][M-2] = x[-1] - offset[i]  (via Q and b)
    - offset_state[i] = dF[i][M-2] + offset[i] = x[-1]  (same for all i)
    """
    num_systems = len(nums_state)
    offsets = np.array(offsets)
    
    # Verify all systems have the same number of states
    if len(set(nums_state)) != 1:
        raise ValueError("All systems must have the same number of states for OffsetMBAR")
    
    M = nums_state[0]  # Number of MBAR states per system
    
    # Total dF dimensions: for each system, M-1 MBAR dFs + 1 offset value = M
    total_dF_dims = num_systems * M
    
    # Independent variables:
    # - For each system: M-2 parameters (first M-2 MBAR dFs)
    # - 1 shared constant c
    num_independent = num_systems * (M - 2) + 1
    
    # Build projection matrix Q and offset vector b
    Q = np.zeros((total_dF_dims, num_independent))
    b = np.zeros(total_dF_dims)
    
    x_idx = 0  # Index in independent variables
    dF_idx = 0  # Index in full dF vector
    
    for i in range(num_systems):
        # First M-2 parameters are independent
        for j in range(M - 2):
            Q[dF_idx + j, x_idx + j] = 1.0
        
        # Last MBAR parameter: dF[i][M-2] = x[-1] - offset[i]
        Q[dF_idx + M - 2, -1] = 1.0  # c from x[-1]
        b[dF_idx + M - 2] = -offsets[i]  # subtract offset[i]
        
        # Offset state: offset_state[i] = dF[i][M-2] + offset[i] = x[-1]
        Q[dF_idx + M - 1, -1] = 1.0
        # b[dF_idx + M - 1] = 0 (already zero)
        
        dF_idx += M
        x_idx += M - 2
    
    return jnp.array(Q), jnp.array(b)

