from collections.abc import Sequence
from typing import List, Tuple
import time
import numpy as np
from numpy import ndarray
import networkx as nx
import jax
import jax.numpy as jnp
from jax import random
from jax import hessian, jit, value_and_grad, vmap
from scipy import optimize
from .bayesmbar import _sample_from_logdensity
from .utils import fmin_newton, _compute_log_likelihood_of_dF

jax.config.update("jax_enable_x64", True)


class CBayesMBAR:
    """
    Coupled BayesMBAR
    """

    def __init__(
        self,
        energies: List[ndarray],
        nums_conf: List[ndarray],
        identical_states: List[List[Tuple[int, int]]],
        sample_size: int = 1000,
        warmup_steps: int = 500,
        method: str = "Newton",
        random_seed: int = None,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        energies : List[ndarray]
            A list of energies for all coupled MBAR systems. The energies should be in the unit of kT.
        nums_conf : List[ndarray]
            A list of the number of configurations for all coupled MBAR systems.
        identical_states : List[List[(int, int)]]
            A list of identical states. Each element in the outer list is a list of tuples and
            represents a group of states that are identical to each other. States are represented
            by a tuple where the first element is the index of a MBAR system and the second element
            is the index of the state in that MBAR system. For example,
            identical_states = [[(0, 3), (1, 0)], [(1, 3), (2, 0), (3, 0)]] means that state 3 in
            system 0 is identical to state 0 in system 1, and state 3 in system 1 is identical to
            state 0 in system 2, and state 0 in system 3.
        sample_size : int, optional
            The number of samples to draw from the likelihood. The default is 1000.
        warmup_steps : int, optional
            The number of warmup steps for the HMC sampler. The default is 500.
        method : str, optional
            The optimization method for finding the mode of the likelihood. Options are "Newton" or "L-BFGS-B". The default is "Newton".
        random_seed : int, optional
            The random seed. The default is None, which means the random seed is generated from the current time.
        verbose : bool, optional
            Whether to print out the progress of the sampling. The default is True.
        """
        self._energies = [jnp.float64(u) for u in energies]
        self._nums_conf = [jnp.int32(n) for n in nums_conf]
        self._identical_states = identical_states

        self._sample_size = sample_size
        self._warmup_steps = warmup_steps

        self._verbose = verbose
        if random_seed is None:
            random_seed = int(time.time())
        self._rng_key = jax.random.PRNGKey(int(time.time()))

        # number of states in each mbar system
        self._nums_state = [len(s) for s in self._nums_conf]

        self._Q = _compute_projection(self._nums_state, identical_states)
        x = jnp.zeros((self._Q.shape[1]))

        print("Solve for the mode of the likelihood")
        if method == "Newton":
            f = jit(value_and_grad(_compute_cmbar_loss_likelihood))
            hess = jit(hessian(_compute_cmbar_loss_likelihood))
            res = fmin_newton(
                f,
                hess,
                x,
                args=(self._Q, self._energies, self._nums_conf),
            )
        elif method == "L-BFGS-B":
            options = {"disp": verbose, "gtol": 1e-8}
            f = jit(value_and_grad(_compute_cmbar_loss_likelihood))
            res = optimize.minimize(
                lambda x: [
                    np.array(r) for r in f(x, self._Q, self._energies, self._nums_conf)
                ],
                x,
                jac=True,
                method="L-BFGS-B",
                tol=1e-12,
                options=options,
            )
        else:
            raise ValueError("Invalid method")

        x_mode_ll = res["x"]
        self._dF_mode_ll = jnp.dot(self._Q, x_mode_ll)
        self._state_F_mode_ll = _dF_to_state_F(self._dF_mode_ll, self._nums_state)

        if self._sample_size > 0:
            print("=====================================================")
            print("Sample from the likelihood")

            self._rng_key, subkey = random.split(self._rng_key)

            def logdensity(x):
                return _compute_cmbar_log_likelihood(
                    x, self._Q, self._energies, self._nums_conf
                )

            self._x_samples_ll = _sample_from_logdensity(
                subkey,
                x_mode_ll,
                logdensity,
                self._warmup_steps,
                self._sample_size,
                self._verbose,
            )
            self._dF_samples_ll = self._x_samples_ll @ self._Q.T

            self._state_F_samples_ll = vmap(_dF_to_state_F, in_axes=[0, None])(
                self._dF_samples_ll, self._nums_state
            )
            self._state_F_mean_ll = [
                jnp.mean(F, axis=0) for F in self._state_F_samples_ll
            ]

    @property
    def F_mode(self) -> List[ndarray]:
        """ The mode of free energies of all states in all MBAR systems. The free energy of 
        state 0 in each system is set to 0.
        """
        return [
            np.array(jax.device_put(F, jax.devices("cpu")[0]))
            for F in self._state_F_mode_ll
        ]

    @property
    def F_samples(self) -> List[ndarray]:
        """ The samples of free energies of all states in all MBAR systems. The free energy of
        state 0 in each system is set to 0.
        """
        return [
            np.array(jax.device_put(F, jax.devices("cpu")[0]))
            for F in self._state_F_samples_ll
        ]

    @property
    def F_mean(self) -> List[ndarray]:
        """ The mean of free energies of all states in all MBAR systems. The free energy of
        state 0 in each system is set to 0.
        """
        return [
            np.array(jax.device_put(F, jax.devices("cpu")[0]))
            for F in self._state_F_mean_ll
        ]

    @property
    def DeltaF_mode(self) -> List[ndarray]:
        """ The mode of free energy differences between all pairs of states in every MBAR system.
        """
        return [F[None, :] - F[:, None] for F in self.F_mode]

    @property
    def DeltaF_mean(self) -> List[ndarray]:
        """ The mean of free energy differences between all pairs of states in every MBAR system.
        """
        return [F[None, :] - F[:, None] for F in self.F_mean]

    @property
    def DeltaF_std(self) -> List[ndarray]:
        """ The standard deviation of free energy differences between all pairs of states in every MBAR system.
        """
        return [np.std(F[:, None, :] - F[:, :, None], 0) for F in self.F_samples]


def _dF_to_state_F(dF: jnp.ndarray, nums_state: List[int]) -> List[jnp.ndarray]:
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
    nums_state: List[int], identical_states: List[List[Tuple[int, int]]]
) -> nx.Graph:
    """
    Generate a graph where each node represents a state and each edge represents a perturbation
    between two states whose free energy difference is to be computed. For states that are identical
    as specified in identical_states, edges are added between all pairs of states that are identical.
    We add edges between all pairs of identical states because we will use them to constraint
    the free energy difference of these edges to be zero.

    Parameters
    ----------
    nums_state : Sequence[int]
        A list of the number of states in each MBAR system.
    identical_states : Sequence[Sequence[(int, int)]]
        Same as the identical_states parameter in the CBayesMBAR class.
    """
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
    nums_state: List[int], identical_states: List[List[Tuple[int, int]]]
) -> ndarray:
    """Compute the projection matrix Q

    This function converts contraints on the free energy differences due to identical states
    to a projection matrix Q that does the following:

    dF = Q @ x, where x is a vector of independent variables without any constraints.
    dF is the vector of all free energy differences between states in all MBAR systems that satisfy
    the constraints because it is generated by the projection matrix Q.

    """

    ## map from state index (i,j), where i is the index of the MBAR system and j is the index of
    ## the state in that system, to the index of the free energy differences in the vector dF
    ## i.e., dF[state_idx_to_dF_idx[(i,j)]] is the free energy difference between state j and
    ## state 0 in system i
    state_dF_idx_to_dF_idx = {}
    idx = 0
    for i in range(len(nums_state)):
        for j in range(1, nums_state[i]):
            state_dF_idx_to_dF_idx[(i, j)] = idx
            idx += 1

    ## build the dF graph and find the cycles in the graph
    graph = _generate_dF_graph(nums_state, identical_states)
    cycles = nx.cycle_basis(graph)

    ## build the matrix A where each row corresponds to a cycle in the graph
    ## For each cycle, we build a vector q that has the same length as the number of free energy
    ## differences.
    ## q[k] = 1 if the k-th free energy difference is in the cycle and the second state
    ## in the edge is the reference state (state 0).
    ## q[k] = -1 if the k-th free energy difference is in the cycle and the first state
    ## in the edge is the reference state.
    ## q[k] = 0 if the k-th free energy difference is not in the cycle.
    ## Note that because all free energy differences are within the same MBAR system,
    ## if a cycle contains an edge between two states in different systems, that edge
    ## must have been added because the two states are identical. In this case, we set q[k] = 0.

    A = []
    for cycle in cycles:
        q = np.zeros(len(state_dF_idx_to_dF_idx))
        for i in range(len(cycle)):
            n1, n2 = cycle[i], cycle[(i + 1) % len(cycle)]

            ## nodes are represented by tuples (m, k), where m is the index of the MBAR system
            ## and k is the index of the state in that system

            ## if the two nodes are in different systems, they must be identical states
            ## and we set q[k] = 0
            if n1[0] != n2[0]:
                continue

            ## check that one of the states is the reference state (state 0)
            assert n1[1] == 0 or n2[1] == 0

            ## set q[k] = 1 if the second state in the edge is the reference state
            ## set q[k] = -1 if the first state in the edge is the reference state
            if n1[1] == 0 and n2[1] != 0:
                q[state_dF_idx_to_dF_idx[n2]] = 1
            elif n1[1] != 0 and n2[1] == 0:
                q[state_dF_idx_to_dF_idx[n1]] = -1
            else:
                raise ValueError("Invalid cycle")

        if np.any(q != 0):
            A.append(q)

    ## compute the projection matrix Q using the QR decomposition
    B = np.array(A).T
    Q, R = np.linalg.qr(B, mode="complete")
    Q = Q[:, B.shape[1] :]

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
