from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

import jax
import jax.numpy as jnp
from jax import random
from jax import hessian, jit, value_and_grad

from .bayesmbar import _compute_log_likelihood_of_F, _sample_dF_from_logdensity
from .utils import fmin_newton

jax.config.update("jax_enable_x64", True)


class BayesCMBAR:
    def __init__(
        self,
        energies: Sequence[npt.NDArray[np.floating]],
        nums_conf: Sequence[npt.NDArray[np.integer]],
        identical_states: Sequence[Sequence[(int, int)]],
        sample_size: int = 1000,
        warmup_steps: int = 500,
        verbose: bool = True,
        random_seed: int = 0,
    ):
        self.energies = [jnp.float64(u) for u in energies]
        self.nums_conf = [jnp.int32(n) for n in nums_conf]
        self.identical_states = identical_states

        self.sample_size = sample_size
        self.warmup_steps = warmup_steps

        self.verbose = verbose
        self.rng_key = jax.random.PRNGKey(random_seed)

        # number of states in each mbar system
        self.nums_state = [len(s) for s in self.nums_conf]

        self.state_idx_to_F_idx = _map_from_state_idx_to_F_idx(
            self.nums_state, identical_states
        )

        self.num_of_free_F = len(set(self.state_idx_to_F_idx.values()))

        dF_init = jnp.zeros((self.num_of_free_F - 1))

        print("Solve for the mode of the likelihood")
        f = jit(value_and_grad(_compute_cmbar_loss_likelihood_of_dF))
        hess = jit(hessian(_compute_cmbar_loss_likelihood_of_dF))
        res = fmin_newton(
            f,
            hess,
            dF_init,
            args=(self.energies, self.nums_conf, self.state_idx_to_F_idx),
        )
        dF = res["x"]
        self._dF_mode_ll = dF

        print("=====================================================")
        print("Sample from the likelihood")

        self.rng_key, subkey = random.split(self.rng_key)

        def logdensity(dF):
            return _compute_cmbar_log_likelihood_of_dF(
                dF, self.energies, self.nums_conf, self.state_idx_to_F_idx
            )

        self._dF_samples_ll = _sample_dF_from_logdensity(
            subkey,
            self._dF_mode_ll,
            logdensity,
            self.warmup_steps,
            self.sample_size,
            self.verbose,
        )

        self._F_mode_ll = jnp.concatenate([jnp.zeros(1), self._dF_mode_ll])
        self._F_samples_ll = jnp.concatenate(
            [jnp.zeros((self.sample_size, 1)), self._dF_samples_ll], axis=1
        )

        self._state_F_mode_ll = _F_to_state_F(
            self._F_mode_ll, self.nums_state, self.state_idx_to_F_idx
        )
        self._state_F_samples_ll = _F_to_state_F(
            self._F_samples_ll, self.nums_state, self.state_idx_to_F_idx
        )

        self._state_F_mean_ll = [jnp.mean(F, axis=0) for F in self._state_F_samples_ll]

    @property
    def F_mode(self):
        return [np.array(jax.device_put(F, jax.devices("cpu")[0])) for F in self._state_F_mode_ll]

    @property
    def F_samples(self):
        return [np.array(jax.device_put(F, jax.devices("cpu")[0])) for F in self._state_F_samples_ll]

    @property
    def F_mean(self):
        return [np.array(jax.device_put(F, jax.devices("cpu")[0])) for F in self._state_F_mean_ll]        

    @property
    def DeltaF_mode(self):
        return [F[None,:] - F[:, None] for F in self.F_mode]
    
    @property
    def DeltaF_mean(self):
        return [F[None,:] - F[:, None] for F in self.F_mean]
    
    @property
    def DeltaF_std(self):
        return [np.std(F[:,None,:] - F[:,:,None], 0) for F in self.F_samples]


def _F_to_state_F(
    F: jnp.ndarray, nums_state: Sequence[int], state_idx_to_F_idx: dict[(int, int), int]
) -> list[jnp.ndarray]:
    F_of_states = []
    for i in range(len(nums_state)):
        idx = jnp.array([state_idx_to_F_idx[(i, j)] for j in range(nums_state[i])])
        F_of_states.append(F[..., idx])
    return F_of_states


def _compute_cmbar_log_likelihood_of_dF(
    dF: jnp.ndarray,
    energies: Sequence[jnp.ndarray],
    nums_conf: Sequence[jnp.ndarray],
    state_idx_to_F_idx: dict[(int, int), int],
):
    nums_state = [len(s) for s in nums_conf]
    F = jnp.concatenate([jnp.zeros(1), dF])
    state_F = _F_to_state_F(F, nums_state, state_idx_to_F_idx)

    log_likelihood = 0
    for energy, num_conf, F in zip(energies, nums_conf, state_F):
        log_likelihood += _compute_log_likelihood_of_F(F, energy, num_conf)

    return log_likelihood


def _compute_cmbar_loss_likelihood_of_dF(
    dF: jnp.ndarray,
    energies: Sequence[jnp.ndarray],
    nums_conf: Sequence[jnp.ndarray],
    state_idx_to_F_idx: dict[(int, int), int],
):
    log_likelihood = _compute_cmbar_log_likelihood_of_dF(
        dF, energies, nums_conf, state_idx_to_F_idx
    )
    N = jnp.sum(jnp.array(nums_conf))
    return -log_likelihood / N


def _map_from_state_idx_to_F_idx(
    nums_state: Sequence[int], identical_states: Sequence[Sequence[(int, int)]]
) -> dict[(int, int), int]:
    ## convert the list of identical states to a dictionary where the key is the state index
    ## and the value is the list of states that are identical to the key state
    iden_states_dict = {}
    for states in identical_states:
        for i in range(len(states)):
            iden_states_dict[states[i]] = [
                states[j] for j in range(len(states)) if j != i
            ]

    ## loop over all states and map each state to an index in the free energy F
    F_idx = 0
    state_idx_to_F_idx = {}
    for i in range(len(nums_state)):
        for j in range(nums_state[i]):
            state_idx = (i, j)

            ## if not other states are identical to this state, assign a new index
            if state_idx not in iden_states_dict.keys():
                state_idx_to_F_idx[state_idx] = F_idx
                F_idx += 1

            ## if there are other states that are identical to this state, go over the list of identical states and check if any of them has been assigned an index. If so, assign the same index to this state. Otherwise, assign a new index
            else:
                for s in iden_states_dict[state_idx]:
                    if s in state_idx_to_F_idx.keys():
                        state_idx_to_F_idx[state_idx] = state_idx_to_F_idx[s]
                        break
                if state_idx not in state_idx_to_F_idx.keys():
                    state_idx_to_F_idx[state_idx] = F_idx
                    F_idx += 1

    return state_idx_to_F_idx
