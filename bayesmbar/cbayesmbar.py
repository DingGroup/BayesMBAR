from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import networkx as nx

import jax
import jax.numpy as jnp
from jax import random
from jax import hessian, jit, value_and_grad, vmap

from .bayesmbar import _compute_log_likelihood_of_dF, _sample_from_logdensity
from .utils import fmin_newton

jax.config.update("jax_enable_x64", True)


class CBayesMBAR:
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

        self.Q = _compute_projection(self.nums_state, identical_states)
        x = jnp.zeros((self.Q.shape[1]))

        print("Solve for the mode of the likelihood")
        f = jit(value_and_grad(_compute_cmbar_loss_likelihood))
        hess = jit(hessian(_compute_cmbar_loss_likelihood))
        res = fmin_newton(
            f,
            hess,
            x,
            args=(self.Q, self.energies, self.nums_conf),
        )

        x_mode_ll = res["x"]
        self._dF_mode_ll = jnp.dot(self.Q, x_mode_ll)
        self._state_F_mode_ll = _dF_to_state_F(self._dF_mode_ll, self.nums_state)

        if self.sample_size > 0:
            print("=====================================================")
            print("Sample from the likelihood")

            self.rng_key, subkey = random.split(self.rng_key)

            def logdensity(x):
                return _compute_cmbar_log_likelihood(
                    x, self.Q, self.energies, self.nums_conf
                )


            self._x_samples_ll = _sample_from_logdensity(
                subkey,
                x_mode_ll,
                logdensity,
                self.warmup_steps,
                self.sample_size,
                self.verbose,
            )
            self._dF_samples_ll = self._x_samples_ll @ self.Q.T

            
            self._state_F_samples_ll = vmap(_dF_to_state_F, in_axes=[0, None])(
                self._dF_samples_ll, self.nums_state
            )
            self._state_F_mean_ll = [jnp.mean(F, axis=0) for F in self._state_F_samples_ll]

    @property
    def F_mode(self):
        return [
            np.array(jax.device_put(F, jax.devices("cpu")[0]))
            for F in self._state_F_mode_ll
        ]

    @property
    def F_samples(self):
        return [
            np.array(jax.device_put(F, jax.devices("cpu")[0]))
            for F in self._state_F_samples_ll
        ]

    @property
    def F_mean(self):
        return [
            np.array(jax.device_put(F, jax.devices("cpu")[0]))
            for F in self._state_F_mean_ll
        ]

    @property
    def DeltaF_mode(self):
        return [F[None, :] - F[:, None] for F in self.F_mode]

    @property
    def DeltaF_mean(self):
        return [F[None, :] - F[:, None] for F in self.F_mean]

    @property
    def DeltaF_std(self):
        return [np.std(F[:, None, :] - F[:, :, None], 0) for F in self.F_samples]


def _dF_to_state_F(dF: jnp.ndarray, nums_state: Sequence[int]) -> list[jnp.ndarray]:
    state_dF = []
    idx = 0
    for n in nums_state:
        state_dF.append(dF[idx : idx + n - 1])
        idx += n - 1
    state_F = [jnp.concatenate([jnp.zeros(1), dF]) for dF in state_dF]
    return state_F


def _F_to_state_F(
    F: jnp.ndarray, nums_state: Sequence[int], state_idx_to_F_idx: dict[(int, int), int]
) -> list[jnp.ndarray]:
    F_of_states = []
    for i in range(len(nums_state)):
        idx = jnp.array([state_idx_to_F_idx[(i, j)] for j in range(nums_state[i])])
        F_of_states.append(F[..., idx])
    return F_of_states


def _compute_cmbar_log_likelihood(
    x: jnp.ndarray,
    Q: jnp.ndarray,
    energies: Sequence[jnp.ndarray],
    nums_conf: Sequence[jnp.ndarray],
):
    dF = jnp.dot(Q, x)
    nums_state = [len(s) for s in nums_conf]
    state_dF = []
    idx = 0
    for n in nums_state:
        state_dF.append(dF[..., idx : idx + n - 1])
        idx += n - 1

    log_likelihood = 0
    for energy, num_conf, dF in zip(energies, nums_conf, state_dF):
        log_likelihood += _compute_log_likelihood_of_dF(dF, energy, num_conf)

    return log_likelihood


def _compute_cmbar_loss_likelihood(
    x: jnp.ndarray,
    Q: jnp.ndarray,
    energies: Sequence[jnp.ndarray],
    nums_conf: Sequence[jnp.ndarray],
):
    log_likelihood = _compute_cmbar_log_likelihood(x, Q, energies, nums_conf)
    N = jnp.sum(jnp.array(nums_conf))
    return -log_likelihood / N


def _generate_dF_graph(
    nums_state: Sequence[int], identical_states: Sequence[Sequence[(int, int)]]
):
    G = nx.Graph()
    for i in range(len(nums_state)):
        for j in range(1, nums_state[i]):
            G.add_edge((i, 0), (i, j))

    for states in identical_states:
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                G.add_edge(states[i], states[j])

    return G


def _compute_projection(
    nums_state: Sequence[int], identical_states: Sequence[Sequence[(int, int)]]
):
    state_dF_idx_to_dF_idx = {}
    idx = 0
    for i in range(len(nums_state)):
        for j in range(1, nums_state[i]):
            state_dF_idx_to_dF_idx[(i, j)] = idx
            idx += 1

    graph = _generate_dF_graph(nums_state, identical_states)
    cycles = nx.cycle_basis(graph)
    A = []

    for cycle in cycles:
        q = np.zeros(len(state_dF_idx_to_dF_idx))
        for i in range(len(cycle)):
            n1, n2 = cycle[i], cycle[(i + 1) % len(cycle)]

            if n1[0] != n2[0]:
                continue

            assert n1[1] == 0 or n2[1] == 0

            if n1[1] == 0 and n2[1] != 0:
                q[state_dF_idx_to_dF_idx[n2]] = 1
            elif n1[1] != 0 and n2[1] == 0:
                q[state_dF_idx_to_dF_idx[n1]] = -1
            else:
                raise ValueError("Invalid cycle")

        if np.any(q != 0):
            A.append(q)

    A = np.array(A).T

    Q, R = np.linalg.qr(A, mode="complete")
    Q = Q[:, A.shape[1] :]
    
    return Q


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
