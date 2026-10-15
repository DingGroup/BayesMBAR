from copy import deepcopy
from functools import partial
from time import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.typing as npt
from jax import jit
from jax.scipy.special import logsumexp

from .utils import (
    _energy_columns,
    _host_batches,
    _loop_over_batches,
    _mbar_weights,
    _num_columns,
    _solve_mbar,
    _solve_mbar_streamed,
)

jax.config.update("jax_enable_x64", True)

__version__ = "1.4.5"

## A small diagonal matrix with __EPS__ as its diagonal elements is added
## to the Hessian matrix to avoid the case where the Hessian matrix is singular
## due to underflow.
__EPS__ = 1e-16


class FastMBAR:
    """
    The FastMBAR class is initialized with an energy matrix and an array
    of num of conformations. The corresponding MBAR equation is solved
    in the constructor. Therefore, the relative free energies of states
    used in the energy matrix is calculated in the constructor. The
    method **calculate_free_energies_for_perturbed_states**
    can be used to calculated the relative free energies of perturbed states.
    """

    def __init__(
        self,
        energy: npt.NDArray[np.float64],
        num_conf: npt.NDArray[np.int64],
        bootstrap: bool = False,
        bootstrap_block_size: int = 3,
        bootstrap_num_rep: int = 100,
        verbose: bool = False,
        method: str = "Newton",
        F_init: npt.NDArray[np.float64] = None,
        batch_size: int | None = None,
        energy_on_device: bool | None = None,
    ) -> None:
        """Initializer for the class FastMBAR

        Args:
            energy (2D ndarray): It has a size of (M x N), where M is the number of states
                and N is the total number of conformations. The entry energy[i,j]
                is the reduced (unitless) energy of conformation j in state i.
                If bootstrapping is used to calculate the uncertainty, the order
                of conformations matters. Conformations sampled from one state
                need to occpy a continuous chunk of collumns. Conformations sampled
                from state k need to occupy collumns to the left of conformations
                sampled from state l if k < l. If bootstrapping is not used, then
                the order of conformation does not matter.
            num_conf (1D int ndarray): It should have a size of M, where num_conf[i] is the num of
                conformations sampled from state i. Therefore, np.sum(num_conf)
                has to be equal to N. All entries in num_conf have to be strictly
                greater than 0.
            bootstrap (bool, optional): If bootstrap is True, the uncertainty of the calculated free energies
                will be estimate using block bootstraping. Defaults to False.
            bootstrap_block_size (int, optional): block size used in block bootstrapping. Defaults to 3.
            bootstrap_num_rep (int, optional): number of repreats in block bootstrapping. Defaults to 100.
            verbose (bool, optional): if verbose is true, the detailed information of solving MBAR equations
                is printed. Defaults to False.
            method (str, optional): the method used to solve the MBAR equation. Defaults to "Newton".
            F_init (1D ndarray, optional): The initial guess for the relative free energies. If None, the initial guess is set to zero. Defaults to None.
            batch_size (int, optional): If given, the MBAR equations are solved and the
                covariance of the free energies is computed by accumulating over batches
                of batch_size conformations; so are the bootstrap replicates and the
                results of calculate_free_energies_of_perturbed_states. The energy
                matrix is copied to the device once, and the temporary memory scales
                with batch_size x M instead of N x M, which lowers the peak device
                memory from about four times the size of the energy matrix to about
                one. With bootstrap, the per-replicate log_prob_mix and resampling
                indices are kept in host memory. Defaults to None, meaning no batching.
            energy_on_device (bool, optional): Only used with batch_size. If True, the
                energy matrix is copied to the device once. If False, it stays in host
                memory and each pass over the conformations streams it to the device in
                batches, so device memory no longer limits the size of the energy
                matrix; this costs one host-to-device copy of the matrix per pass and
                is currently supported only for the Newton method without bootstrap or
                perturbed states. If None, the energy matrix is kept on the device when
                it fits in the free device memory with some margin, and streamed
                otherwise. Defaults to None.
        """

        #### check the parameters: energy and num_conf
        ## Note that both self.energy and self.conf will be on CPUs no matter
        ## if cuda is True or False.

        ## check energy
        if isinstance(energy, (np.ndarray, jnp.ndarray)):
            if energy.ndim != 2:
                raise ValueError("energy has to be two dimensional")
            self.energy = energy.astype(np.float64)
        else:
            raise TypeError("energy has to be a 2-D ndarray")

        ## check num_conf
        if isinstance(num_conf, (np.ndarray, jnp.ndarray)):
            if num_conf.ndim != 1:
                raise ValueError("num_conf has to be one dimensional")
            self.num_conf = num_conf.astype(np.float64)
        else:
            raise TypeError("num_conf has to be a 1-D ndarray.")

        ## check the shape of energy and num_conf
        if energy.shape[0] != num_conf.shape[0]:
            raise ValueError(
                "The number of rows in energy must be equal to the length of num_conf. "
                f"Got {energy.shape[0]} rows in energy and {num_conf.shape[0]} elements in num_conf."
            )

        ## check if the number of conformations sampled from each state is greater than 0
        if np.any(self.num_conf <= 0):
            raise ValueError(
                "all entries in num_conf have to be strictly greater than 0."
            )

        ## check if the total number of conformations is equal to the number of columns in energy
        total_num_conf = np.sum(self.num_conf)
        num_columns = self.energy.shape[1]
        if total_num_conf != num_columns:
            raise ValueError(
                f"The sum of num_conf ({total_num_conf}) must be equal to the number of columns in energy ({num_columns})."
            )

        self.M = self.energy.shape[0]
        self.N = self.energy.shape[1]

        ## whether to use bootstrap to estimate the uncertainty of the calculated free energies
        ## bootstrap needs to be a boolean
        if not isinstance(bootstrap, bool):
            raise TypeError("bootstrap has to be a boolean.")
        self.bootstrap = bootstrap

        ## block size used in block bootstrapping
        if not isinstance(bootstrap_block_size, int):
            raise TypeError("bootstrap_block_size has to be an integer.")
        self.bootstrap_block_size = bootstrap_block_size

        ## number of repeats in block bootstrapping
        if not isinstance(bootstrap_num_rep, int):
            raise TypeError("bootstrap_num_rep has to be an integer.")
        self.bootstrap_num_rep = bootstrap_num_rep

        ## whether to print the detailed information of solving MBAR equations
        ## verbose needs to be a boolean
        if not isinstance(verbose, bool):
            raise TypeError("verbose has to be a boolean.")
        self.verbose = verbose

        ## method used to solve the MBAR equation
        if not isinstance(method, str):
            raise TypeError("method has to be a string.")
        if method not in ["Newton", "L-BFGS"]:
            raise ValueError("method has to be Newton or L-BFGS.")
        self.method = method

        ## batch size used to solve the MBAR equation and compute the covariance
        if batch_size is not None:
            if not isinstance(batch_size, int) or isinstance(batch_size, bool):
                raise TypeError("batch_size has to be an integer or None.")
            if batch_size <= 0:
                raise ValueError("batch_size has to be strictly greater than 0.")
        self.batch_size = batch_size

        ## whether the energy matrix is kept on the device or streamed from the host
        if energy_on_device is not None and not isinstance(energy_on_device, bool):
            raise TypeError("energy_on_device has to be a boolean or None.")
        if batch_size is None:
            if energy_on_device is False:
                raise ValueError("energy_on_device=False requires batch_size.")
            energy_on_device = True
        elif energy_on_device is None:
            energy_on_device = _fits_on_device(self.energy.nbytes)
        if not energy_on_device:
            unsupported = []
            if self.method != "Newton":
                unsupported.append(f'method="{self.method}"')
            if self.bootstrap:
                unsupported.append("bootstrap")
            if unsupported:
                raise NotImplementedError(
                    "Streaming the energy matrix from host memory "
                    "(energy_on_device=False, or chosen automatically because it "
                    "does not fit in device memory) is not yet supported with "
                    + " and ".join(unsupported)
                    + '. Use method="Newton" and bootstrap=False, or '
                    "energy_on_device=True."
                )
        self.energy_on_device = energy_on_device

        # ## solve the MBAR equation
        if self.bootstrap is False and not self.energy_on_device:
            ## keep the energy matrix in host memory, transposed to (N, M) so that
            ## each batch of conformations is a contiguous block of rows
            energy_T = np.ascontiguousarray(self.energy.T)
            num_conf = jnp.asarray(self.num_conf)

            dF_init = jnp.zeros(self.M - 1)
            if F_init is not None:
                dF_init = F_init[1:] - F_init[0]
            dF = _solve_mbar_streamed(
                dF_init, energy_T, num_conf, self.verbose, self.batch_size
            )

            ## shift self._F such that \sum_i F[i]*num_conf[i] = 0
            F = jnp.concat([jnp.zeros(1), dF])
            pi = num_conf / self.N
            self._F = F - jnp.sum(pi * F)

            self._log_prob_mix, gram = _log_prob_mix_and_gram_streamed(
                self._F, energy_T, num_conf, self.batch_size
            )
            del energy_T
            self._log_mu = -self.log_prob_mix

            self.P = None
            self._F_cov = _covariance_from_gram(gram, num_conf, rtol=1e-7)
            self._F_std = jnp.sqrt(self._F_cov.diagonal())

            self._DeltaF = self._F[None, :] - self._F[:, None]
            self._DeltaF_cov = (
                self._F_cov.diagonal()[:, None]
                + self._F_cov.diagonal()[None, :]
                - 2 * self._F_cov
            )
            self._DeltaF_std = jnp.sqrt(self._DeltaF_cov)

        elif self.bootstrap is False and self.batch_size is not None:
            ## copy the energy matrix to the device once; everything else is
            ## accumulated over batches of conformations. jax.device_put copies
            ## directly, while jnp.asarray briefly holds two copies on the device.
            energy = jax.device_put(self.energy)
            num_conf = jnp.asarray(self.num_conf)

            dF_init = jnp.zeros(self.M - 1)
            if F_init is not None:
                dF_init = F_init[1:] - F_init[0]
            dF = _solve_mbar(
                dF_init,
                energy,
                num_conf,
                self.method,
                verbose=self.verbose,
                batch_size=self.batch_size,
            )

            ## shift self._F such that \sum_i F[i]*num_conf[i] = 0
            F = jnp.concat([jnp.zeros(1), dF])
            pi = num_conf / self.N
            self._F = F - jnp.sum(pi * F)

            self._log_prob_mix, gram = _log_prob_mix_and_gram_in_batches(
                self._F, energy, num_conf, self.batch_size
            )
            self._log_mu = -self.log_prob_mix

            ## same covariance as the unbatched branch below, computed from the
            ## M x M Gram matrix P^T P instead of the N x M matrix P
            self.P = None
            self._F_cov = _covariance_from_gram(gram, num_conf, rtol=1e-7)

            self._F_std = jnp.sqrt(self._F_cov.diagonal())

            self._DeltaF = self._F[None, :] - self._F[:, None]
            self._DeltaF_cov = (
                self._F_cov.diagonal()[:, None]
                + self._F_cov.diagonal()[None, :]
                - 2 * self._F_cov
            )
            self._DeltaF_std = jnp.sqrt(self._DeltaF_cov)

        elif self.bootstrap is False:
            dF_init = jnp.zeros(self.M - 1)
            if F_init is not None:
                dF_init = F_init[1:] - F_init[0]
            dF = _solve_mbar(
                dF_init,
                self.energy,
                self.num_conf,
                self.method,
                verbose=self.verbose,
            )

            ## shift self._F such that \sum_i F[i]*num_conf[i] = 0
            F = jnp.concat([jnp.zeros(1), dF])
            pi = self.num_conf / self.N
            self._F = F - jnp.sum(pi * F)

            b = -self._F - jnp.log(self.num_conf)
            self._log_prob_mix = logsumexp(-(self.energy + b[:, None]), axis=0)
            self._log_mu = -self.log_prob_mix

            ## Compute self._F_cov under the constraint that \sum_i F[i]*num_conf[i] = 0

            ## There are two ways to compute the covariance matrix of self._F
            ## See equation 4.2 and 6.4 in the following paper for details
            ## "Kong, A., et al. "A theory of statistical models for Monte Carlo integration."
            ## Journal of the Royal Statistical Society Series B: Statistical Methodology 65.3
            ## (2003): 585-604." https://doi.org/10.1111/1467-9868.00404

            ## The first way as shown in the following uses equation 4.2
            ## this method is more general than the following method, meaning that
            ## it can also be used to compute covariance matrix for perturbed states.
            ## Therefore it is used here.

            self.P = jnp.exp(
                -(self.energy - self._F[:, None] + self._log_prob_mix)
            ).transpose()
            W = jnp.diag(self.num_conf)
            Q, R = jnp.linalg.qr(self.P)
            A = jnp.eye(self.M, device=W.device) - R @ W @ R.T
            self._F_cov = R.T @ jnp.linalg.pinv(A, hermitian=True, rtol=1e-7) @ R

            # # The second way uses equation 6.4
            # if self._batch_size is None:
            #     H = _compute_hessian_log_likelihood_of_F(
            #         self._F, self.energy, self.num_conf
            #     )
            # else:
            #     H = _compute_hessian_log_likelihood_of_F_in_batch(
            #         self._F, self.energy, self.num_conf, self._batch_size
            #     )
            # Hp = H.new_zeros((self.M + 1, self.M + 1)).cpu()
            # Hp[0 : self.M, 0 : self.M] = H
            # Hp[self.M, 0 : self.M] = -self.num_conf
            # Hp[0 : self.M, self.M] = -self.num_conf
            # Hp[self.M, self.M] = 0
            # self._F_cov = torch.linalg.inv(-Hp)[0 : self.M, 0 : self.M]
            # self._F_cov = self._F_cov - torch.diag(1 / self.num_conf) + 1 / self.N

            self._F_std = jnp.sqrt(self._F_cov.diagonal())

            self._DeltaF = self._F[None, :] - self._F[:, None]
            self._DeltaF_cov = (
                self._F_cov.diagonal()[:, None]
                + self._F_cov.diagonal()[None, :]
                - 2 * self._F_cov
            )
            self._DeltaF_std = jnp.sqrt(self._DeltaF_cov)

        elif self.bootstrap is True:
            dFs = []
            log_prob_mix = []
            dF_init = jnp.zeros(self.M - 1)
            if F_init is not None:
                dF_init = F_init[1:] - F_init[0]
            self._conf_idx = []
            if self.batch_size is not None:
                ## copy the energy matrix to the device once; each resample is
                ## gathered batch by batch instead of forming energy[:, conf_idx]
                energy = jax.device_put(self.energy)
                num_conf_dev = jnp.asarray(self.num_conf)
            ## seed once and split per replicate so that replicates drawn within
            ## the same millisecond still get different resamples
            key = jr.PRNGKey(int(time() * 1000))
            for _ in range(self.bootstrap_num_rep):
                key, subkey = jr.split(key)
                conf_idx = _bootstrap_conf_idx(
                    subkey, num_conf, self.bootstrap_block_size
                )
                if self.batch_size is None:
                    dF = _solve_mbar(
                        dF_init,
                        self.energy[:, conf_idx],
                        self.num_conf,
                        self.method,
                        verbose=self.verbose,
                    )
                else:
                    dF = _solve_mbar(
                        dF_init,
                        energy,
                        num_conf_dev,
                        self.method,
                        verbose=self.verbose,
                        batch_size=self.batch_size,
                        conf_idx=conf_idx,
                    )
                dF_init = dF
                dFs.append(deepcopy(dF))

                F = jnp.concat([jnp.zeros(1), dF])
                ## shift F such that \sum_i F[i]*num_conf[i] = 0
                pi = self.num_conf / self.N
                F = F - jnp.sum(pi * F)

                ## same definition as in the non-bootstrap branch; using
                ## log(num_conf / N) here shifted the free energies of perturbed
                ## states by -log(N)
                if self.batch_size is None:
                    b = -F - jnp.log(self.num_conf)
                    log_prob_mix.append(
                        logsumexp(-(self.energy[:, conf_idx] + b[:, None]), axis=0)
                    )
                    self._conf_idx.append(conf_idx)
                else:
                    lpm, _ = _log_prob_mix_and_gram_in_batches(
                        F,
                        energy,
                        num_conf_dev,
                        self.batch_size,
                        conf_idx,
                        with_gram=False,
                    )
                    ## keep the per-replicate arrays (each of length N) in host
                    ## memory so that device memory does not grow with the
                    ## number of replicates
                    log_prob_mix.append(np.asarray(lpm))
                    self._conf_idx.append(np.asarray(conf_idx))

            dF = jnp.stack(dFs, axis=1)
            F = jnp.concat([jnp.zeros((1, dF.shape[1])), dF], axis=0)

            ## shift F such that \sum_i F[i]*num_conf[i] = 0
            pi = self.num_conf / self.N
            self._F_bootstrap = F - jnp.sum(pi[:, None] * F, axis=0)

            self._F = jnp.mean(self._F_bootstrap, axis=1)
            self._F_std = jnp.std(self._F_bootstrap, axis=1)
            self._F_cov = jnp.cov(self._F_bootstrap)

            if self.batch_size is None:
                self._log_prob_mix = jnp.stack(log_prob_mix, axis=0)
            else:
                self._log_prob_mix = np.stack(log_prob_mix, axis=0)
            DeltaF = self._F_bootstrap[None, :, :] - self._F_bootstrap[:, None, :]
            self._DeltaF = jnp.mean(DeltaF, axis=2)
            self._DeltaF_std = jnp.std(DeltaF, axis=2)

    @property
    def F(self) -> np.ndarray:
        r"""Free energies of the states under the constraint :math:`\sum_{k=1}^{M} N_k * F_k = 0`,
        where :math:`N_k` is the number of conformations sampled from state k.
        """
        return np.array(self._F)

    @property
    def F_std(self) -> np.ndarray:
        r"""Standard deviation of the free energies of the states under the constraint
        :math:`\sum_{k=1}^{M} N_k * F_k = 0`,
        where :math:`N_k` is the number of conformations sampled from state k.
        """
        return np.array(self._F_std)

    @property
    def F_cov(self) -> np.ndarray:
        r"""Covariance matrix of the free energies of the states under the constraint
        :math:`\sum_{k=1}^{M} N_k * F_k = 0`,
        where :math:`N_k` is the number of conformations sampled from state k.
        """
        return np.array(self._F_cov)

    @property
    def DeltaF(self) -> np.ndarray:
        r"""Free energy difference between states.
        :math:`\mathrm{DeltaF}[i,j]` is the free energy difference between state j and state i,
        i.e., :math:`\mathrm{DeltaF}[i,j] = F[j] - F[i]` .
        """
        return np.array(self._DeltaF)

    @property
    def DeltaF_std(self) -> np.ndarray:
        r"""Standard deviation of the free energy difference between states.
        :math:`\mathrm{DeltaF_std}[i,j]` is the standard deviation of the free energy
        difference :math:`\mathrm{DeltaF}[i,j]`.
        """
        return np.array(self._DeltaF_std)

    @property
    def log_prob_mix(self) -> np.ndarray:
        """the log probability density of conformations in the mixture distribution."""
        return np.array(self._log_prob_mix)

    def calculate_free_energies_of_perturbed_states(
        self, energy_perturbed: npt.NDArray[np.float64]
    ) -> dict:
        r"""calculate free energies for perturbed states.

        Args:
            energy_perturbed (2-D float ndarray): A matrix with size of (L, N). Each row of the
                energy_perturbed matrix represents a state and the value energy_perturbed[l,n]
                represents the reduced energy of the n'th conformation in the l'th perturbed state.

        Returns:
            dict: a dictionary containing the following keys:

                - **F** - the free energy of the perturbed states.
                - **F_std** - the standard deviation of the free energy of the perturbed states.
                - **F_cov** - the covariance between the free energies of the perturbed states.
                - **DeltaF** - $\mathrm{DeltaF}[k,l]$ is the free energy difference between state
                  $k$ and state $l$, i.e., $\mathrm{DeltaF}[k,l] = F[l] - F[k]$.
                - **DeltaF_std** - the standard deviation of the free energy difference.
        """

        if isinstance(energy_perturbed, np.ndarray):
            energy_perturbed = energy_perturbed.astype(np.float64)
        else:
            raise TypeError("energy_perturbed has to be a numpy array")

        if energy_perturbed.ndim != 2:
            raise ValueError(
                "energy_perturbed has to be a 2-D ndarray or a 2-D tensor."
            )
        if energy_perturbed.shape[1] != self.energy.shape[1]:
            raise ValueError(
                "the number of columns in energy_perturbed has to be equal to the number of columns in energy."
            )

        L = energy_perturbed.shape[0]

        if not self.energy_on_device:
            raise NotImplementedError(
                "calculate_free_energies_of_perturbed_states is not yet supported "
                "when the energy matrix is streamed from host memory "
                "(energy_on_device=False)."
            )

        F = None
        F_cov = None
        F_std = None

        if not self.bootstrap and self.batch_size is not None:
            ## same results as the unbatched branch below, accumulated over batches
            ## of conformations; the N x (M + L) matrix P is never formed
            energy = jax.device_put(self.energy)
            energy_perturbed = jax.device_put(energy_perturbed)
            F = _perturbed_F_in_batches(
                energy_perturbed, self._log_prob_mix, self.batch_size
            )
            gram = _extended_gram_in_batches(
                self._F,
                F,
                energy,
                energy_perturbed,
                self._log_prob_mix,
                self.batch_size,
            )
            weights = jnp.concat([jnp.asarray(self.num_conf), jnp.zeros(L)])
            F_cov = _covariance_from_gram(gram, weights, rtol=1e-8)

            F_cov = F_cov[self.M :, self.M :]
            F_std = jnp.sqrt(F_cov.diagonal())

            DeltaF = F[None, :] - F[:, None]
            DeltaF_cov = (
                F_cov.diagonal()[:, None] + F_cov.diagonal()[None, :] - 2 * F_cov
            )
            DeltaF_std = jnp.sqrt(DeltaF_cov)

        elif not self.bootstrap:
            du = energy_perturbed + self._log_prob_mix
            F = -logsumexp(-du, axis=1)

            F_ext = jnp.concat([self._F, F])
            U = jnp.concat([self.energy, energy_perturbed], axis=0).transpose()
            self._P = jnp.exp(-(U - F_ext + self._log_prob_mix[:, None]))

            W = jnp.diag(jnp.concat([self.num_conf, jnp.zeros(L)]))
            Q, R = jnp.linalg.qr(self._P)
            A = jnp.eye(self.M + L, device=W.device) - R @ W @ R.T
            F_cov = R.T @ jnp.linalg.pinv(A, hermitian=True, rtol=1e-8) @ R

            F_cov = F_cov[self.M :, self.M :]
            F_std = jnp.sqrt(F_cov.diagonal())

            DeltaF = F[None, :] - F[:, None]
            DeltaF_cov = (
                F_cov.diagonal()[:, None] + F_cov.diagonal()[None, :] - 2 * F_cov
            )
            DeltaF_std = jnp.sqrt(DeltaF_cov)

        else:
            F_list = []
            if self.batch_size is not None:
                energy_perturbed = jax.device_put(energy_perturbed)
            for k in range(self.bootstrap_num_rep):
                if self.batch_size is None:
                    du = energy_perturbed[:, self._conf_idx[k]] + self._log_prob_mix[k]
                    F = -logsumexp(-du, axis=1)
                else:
                    F = _perturbed_F_in_batches(
                        energy_perturbed,
                        jax.device_put(self._log_prob_mix[k]),
                        self.batch_size,
                        jax.device_put(self._conf_idx[k]),
                    )
                F_list.append(F)
            Fs = jnp.stack(F_list, axis=1)
            F = jnp.mean(Fs, axis=1)
            F_std = jnp.std(Fs, axis=1)
            F_cov = jnp.cov(Fs)

            DeltaF = F[None, :] - F[:, None]
            DeltaF_std = jnp.std(Fs[:, :, None] - Fs[:, None, :], axis=1)

        results = {
            "F": np.array(F),
            "F_std": np.array(F_std),
            "F_cov": np.array(F_cov),
            "DeltaF": np.array(DeltaF),
            "DeltaF_std": np.array(DeltaF_std),
        }

        return results





@partial(jit, static_argnames=["batch_size", "with_gram"])
def _log_prob_mix_and_gram_in_batches(
    F, energy, num_conf, batch_size, conf_idx=None, with_gram=True
):
    """Compute log_prob_mix and the Gram matrix P^T P over batches of conformations.

    log_prob_mix[n] = logsumexp_k(F_k + log N_k - u_k(x_n)) is the log density of
    conformation n in the mixture distribution, and
    P[n, k] = exp(F_k - u_k(x_n) - log_prob_mix[n]) = p[n, k] / N_k, where p[n] are
    the MBAR weights of conformation n. P^T P is accumulated as
    sum_n p[n]^T p[n] / (N_k N_l), so the N x M matrix P is never formed. If
    conf_idx is given, the energy matrix is energy[:, conf_idx]. If with_gram is
    False, the returned Gram matrix is zero.
    """
    M = energy.shape[0]
    N = _num_columns(energy, conf_idx)
    logn = jnp.log(num_conf)

    def body(start, size, carry):
        log_prob_mix, ptp = carry
        u = _energy_columns(energy, conf_idx, start, size)
        logits = F + logn - u.T
        lse = logsumexp(logits, axis=1)
        log_prob_mix = jax.lax.dynamic_update_slice_in_dim(
            log_prob_mix, lse, start, axis=0
        )
        if with_gram:
            p = jnp.exp(logits - lse[:, None])
            ptp = ptp + p.T @ p
        return log_prob_mix, ptp

    init = (jnp.zeros((N,), energy.dtype), jnp.zeros((M, M), energy.dtype))
    log_prob_mix, ptp = _loop_over_batches(body, N, batch_size, init)
    gram = ptp / (num_conf[:, None] * num_conf[None, :])
    return log_prob_mix, gram


@partial(jit, static_argnames=["batch_size"])
def _perturbed_F_in_batches(energy_perturbed, log_prob_mix, batch_size, conf_idx=None):
    """F_l = -logsumexp_n(-(u_l(x_n) + log_prob_mix[n])) accumulated over batches.

    If conf_idx is given (a bootstrap resample), the columns of energy_perturbed are
    energy_perturbed[:, conf_idx], and log_prob_mix must already be aligned with
    the resampled columns.
    """
    L = energy_perturbed.shape[0]

    def body(start, size, acc):
        u = _energy_columns(energy_perturbed, conf_idx, start, size)
        lpm = jax.lax.dynamic_slice_in_dim(log_prob_mix, start, size)
        return jnp.logaddexp(acc, logsumexp(-(u + lpm), axis=1))

    init = jnp.full((L,), -jnp.inf, energy_perturbed.dtype)
    return -_loop_over_batches(body, log_prob_mix.shape[0], batch_size, init)


@partial(jit, static_argnames=["batch_size"])
def _extended_gram_in_batches(
    F, F_perturbed, energy, energy_perturbed, log_prob_mix, batch_size
):
    """P^T P over the M sampled and L perturbed states, accumulated over batches,
    where P[n, k] = exp(F_k - u_k(x_n) - log_prob_mix[n])."""
    F_ext = jnp.concatenate([F, F_perturbed])
    K = F_ext.shape[0]

    def body(start, size, acc):
        u = jnp.concatenate(
            [
                jax.lax.dynamic_slice_in_dim(energy, start, size, axis=1),
                jax.lax.dynamic_slice_in_dim(energy_perturbed, start, size, axis=1),
            ],
            axis=0,
        )
        lpm = jax.lax.dynamic_slice_in_dim(log_prob_mix, start, size)
        P = jnp.exp(F_ext - u.T - lpm[:, None])
        return acc + P.T @ P

    init = jnp.zeros((K, K), energy.dtype)
    return _loop_over_batches(body, log_prob_mix.shape[0], batch_size, init)


@jit
def _add_gram_batch(ptp, F, logn, u_T):
    lse, p = _mbar_weights(F, logn, u_T)
    return ptp + p.T @ p, lse


def _log_prob_mix_and_gram_streamed(F, energy_T, num_conf, batch_size):
    """Same as _log_prob_mix_and_gram_in_batches, streaming the transposed energy
    matrix energy_T (a host (N, M) array) to the device in batches."""
    logn = jnp.log(num_conf)
    M = energy_T.shape[1]
    ptp = jnp.zeros((M, M))
    lse_batches = []
    for u_T in _host_batches(energy_T, batch_size):
        ## block so that at most the current and the prefetched batch are on the
        ## device; see _host_batches
        ptp, lse = jax.block_until_ready(_add_gram_batch(ptp, F, logn, u_T))
        lse_batches.append(lse)
    log_prob_mix = jnp.concatenate(lse_batches)
    gram = ptp / (num_conf[:, None] * num_conf[None, :])
    return log_prob_mix, gram


def _fits_on_device(nbytes, margin=1.25):
    """Whether an array of nbytes, plus margin for temporaries, fits in the free
    memory of the default device. Devices that do not report memory statistics
    (e.g. the CPU) are assumed to fit."""
    stats = jax.devices()[0].memory_stats()
    if not stats or "bytes_limit" not in stats:
        return True
    free = stats["bytes_limit"] - stats["bytes_in_use"]
    return nbytes * margin <= free


def _covariance_from_gram(gram, weights, rtol):
    """Covariance of equation 4.2 of Kong et al. (2003) from the Gram matrix P^T P.

    The unbatched code computes R.T @ pinv(I - R W R^T) @ R with R from the QR
    decomposition of P. The result depends on R only through R^T R = P^T P, so any
    R with R^T R = P^T P gives the same covariance. R is built here from the
    eigendecomposition of P^T P, so P itself is never needed. weights is the
    diagonal of W.
    """
    eigval, eigvec = jnp.linalg.eigh(gram)
    R = jnp.sqrt(jnp.clip(eigval, 0.0))[:, None] * eigvec.T
    A = jnp.eye(gram.shape[0]) - R @ jnp.diag(weights) @ R.T
    return R.T @ jnp.linalg.pinv(A, hermitian=True, rtol=rtol) @ R


def _bootstrap_conf_idx(key, num_conf, bootstrap_block_size):
    num_conf_cumsum = np.cumsum(num_conf).tolist()
    num_conf_cumsum.pop(-1)
    num_conf_cumsum = [0] + num_conf_cumsum
    conf_idx = []

    for i in range(len(num_conf)):
        len_seq = int(num_conf[i])  ## using circular block bootstrap
        num_sample_block = -(-len_seq // bootstrap_block_size)
        key, subkey = jr.split(key)
        idxs = jr.randint(subkey, (num_sample_block,), 0, len_seq)
        ## the start index of each block plus offsets within a block, flattened
        sample_idx = (idxs[:, None] + jnp.arange(bootstrap_block_size)).reshape(-1)
        sample_idx = jnp.remainder(sample_idx, len_seq)
        sample_idx = sample_idx[0:len_seq]
        sample_idx = sample_idx + num_conf_cumsum[i]
        conf_idx.append(sample_idx)
    conf_idx = jnp.concat(conf_idx)
    return conf_idx
