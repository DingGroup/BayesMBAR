from copy import deepcopy
from functools import partial
from time import time
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from beartype import beartype
from jax import jit
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float, Int, PRNGKeyArray, Real, jaxtyped

from .utils import (
    _energy_columns,
    _host_batches,
    _loop_over_batches,
    _mbar_weights,
    _num_columns,
    _rows,
    _solve_mbar,
    _solve_mbar_streamed,
)

jax.config.update("jax_enable_x64", True)

__version__ = "1.4.5"

## A small diagonal matrix with __EPS__ as its diagonal elements is added
## to the Hessian matrix to avoid the case where the Hessian matrix is singular
## due to underflow.
__EPS__ = 1e-16

## Shapes in the jaxtyping annotations below:
##   M: number of states            
##   N: number of conformations
##   L: number of perturbed states 
##   B: number of conformations in a batch
##   R: number of bootstrap replicates
## The public entry points accept NumPy or JAX arrays and are type-checked at
## runtime (jaxtyping with beartype); the private helpers are annotated only.
HostOrDeviceArray = np.ndarray | Array


class FastMBAR:
    """
    The FastMBAR class is initialized with an energy matrix and an array
    of num of conformations. The corresponding MBAR equation is solved
    in the constructor. Therefore, the relative free energies of states
    used in the energy matrix is calculated in the constructor. The
    method **calculate_free_energies_for_perturbed_states**
    can be used to calculated the relative free energies of perturbed states.
    """

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        energy: Real[HostOrDeviceArray, "M N"],
        num_conf: Real[HostOrDeviceArray, " M"],
        bootstrap: bool = False,
        bootstrap_block_size: int = 3,
        bootstrap_num_rep: int = 100,
        verbose: bool = False,
        method: Literal["Newton", "L-BFGS"] = "Newton",
        F_init: Real[HostOrDeviceArray, " M"] | None = None,
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
                matrix; this costs one host-to-device copy of the matrix per pass.
                Newton usually needs the fewest passes; L-BFGS needs more passes but
                each skips the M x M Hessian, which can pay off when the number of
                states M is large. In this mode the
                energy matrix is stored transposed, as a contiguous (N, M) array; pass
                a float64 Fortran-ordered energy (e.g. energy_T.T for a C-ordered
                (N, M) array energy_T, or a np.memmap opened with order="F") to avoid
                a copy. If None, the energy matrix is kept on the device when it fits
                in the free device memory with some margin, and streamed otherwise.
                Defaults to None.

        Raises:
            jaxtyping.TypeCheckError: (a subclass of TypeError) if an argument has the
                wrong type, dtype kind or shape, e.g. if energy is not 2-D or
                num_conf does not have one entry per row of energy.
            ValueError: if num_conf has a non-positive entry or does not sum to the
                number of columns of energy, or if batch_size is not positive.
        """

        ## The types and shapes of the arguments are checked by jaxtyping and
        ## beartype (see the annotations); only the checks that cannot be expressed
        ## as annotations are done here.
        self.num_conf = num_conf.astype(np.float64)

        ## check if the number of conformations sampled from each state is greater than 0
        if np.any(self.num_conf <= 0):
            raise ValueError(
                "all entries in num_conf have to be strictly greater than 0."
            )

        ## check if the total number of conformations is equal to the number of columns in energy
        total_num_conf = np.sum(self.num_conf)
        num_columns = energy.shape[1]
        if total_num_conf != num_columns:
            raise ValueError(
                f"The sum of num_conf ({total_num_conf}) must be equal to the number of columns in energy ({num_columns})."
            )

        self.M = energy.shape[0]
        self.N = energy.shape[1]

        self.bootstrap = bootstrap
        self.bootstrap_block_size = bootstrap_block_size
        self.bootstrap_num_rep = bootstrap_num_rep
        self.verbose = verbose
        self.method = method

        ## batch size used to solve the MBAR equation and compute the covariance;
        ## bool is a subclass of int, so the annotation alone does not reject True
        if batch_size is not None:
            if isinstance(batch_size, bool):
                raise TypeError("batch_size has to be an integer or None.")
            if batch_size <= 0:
                raise ValueError("batch_size has to be strictly greater than 0.")
        self.batch_size = batch_size

        ## whether the energy matrix is kept on the device or streamed from the host
        if batch_size is None:
            if energy_on_device is False:
                raise ValueError("energy_on_device=False requires batch_size.")
            energy_on_device = True
        elif energy_on_device is None:
            energy_on_device = _fits_on_device(self.M * self.N * 8)
        self.energy_on_device = energy_on_device

        if self.energy_on_device:
            self.energy = energy.astype(np.float64)
        else:
            ## Keep only a contiguous (N, M) copy so that each batch of
            ## conformations is a contiguous block of rows; self.energy is a view
            ## of it. No copy is made if energy is already float64 and
            ## Fortran-ordered (e.g. energy_T.T for a C-ordered (N, M) array, or a
            ## np.memmap opened with order="F").
            self._energy_T = np.ascontiguousarray(np.asarray(energy).T, dtype=np.float64)
            self.energy = self._energy_T.T

        # ## solve the MBAR equation
        if self.bootstrap is False and not self.energy_on_device:
            energy_T = self._energy_T
            num_conf = jnp.asarray(self.num_conf)

            dF_init = jnp.zeros(self.M - 1)
            if F_init is not None:
                dF_init = F_init[1:] - F_init[0]
            dF = _solve_mbar_streamed(
                dF_init,
                energy_T,
                num_conf,
                self.verbose,
                self.batch_size,
                method=self.method,
            )

            ## shift self._F such that \sum_i F[i]*num_conf[i] = 0
            F = jnp.concat([jnp.zeros(1), dF])
            pi = num_conf / self.N
            self._F = F - jnp.sum(pi * F)

            ## log_prob_mix stays in host memory, like the energy matrix
            self._log_prob_mix, gram = _log_prob_mix_and_gram_streamed(
                self._F, energy_T, num_conf, self.batch_size
            )
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
                num_conf_dev = jnp.asarray(self.num_conf)
            if self.batch_size is not None and self.energy_on_device:
                ## copy the energy matrix to the device once; each resample is
                ## gathered batch by batch instead of forming energy[:, conf_idx]
                energy = jax.device_put(self.energy)
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
                elif self.energy_on_device:
                    dF = _solve_mbar(
                        dF_init,
                        energy,
                        num_conf_dev,
                        self.method,
                        verbose=self.verbose,
                        batch_size=self.batch_size,
                        conf_idx=conf_idx,
                    )
                else:
                    ## the resampled rows are gathered on the host batch by batch
                    conf_idx = np.asarray(conf_idx)
                    dF = _solve_mbar_streamed(
                        dF_init,
                        self._energy_T,
                        num_conf_dev,
                        self.verbose,
                        self.batch_size,
                        conf_idx,
                        method=self.method,
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
                    if self.energy_on_device:
                        lpm, _ = _log_prob_mix_and_gram_in_batches(
                            F,
                            energy,
                            num_conf_dev,
                            self.batch_size,
                            conf_idx,
                            with_gram=False,
                        )
                    else:
                        lpm, _ = _log_prob_mix_and_gram_streamed(
                            F,
                            self._energy_T,
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
    def F(self) -> Float[np.ndarray, " M"]:
        r"""Free energies of the states under the constraint :math:`\sum_{k=1}^{M} N_k * F_k = 0`,
        where :math:`N_k` is the number of conformations sampled from state k.
        """
        return np.array(self._F)

    @property
    def F_std(self) -> Float[np.ndarray, " M"]:
        r"""Standard deviation of the free energies of the states under the constraint
        :math:`\sum_{k=1}^{M} N_k * F_k = 0`,
        where :math:`N_k` is the number of conformations sampled from state k.
        """
        return np.array(self._F_std)

    @property
    def F_cov(self) -> Float[np.ndarray, "M M"]:
        r"""Covariance matrix of the free energies of the states under the constraint
        :math:`\sum_{k=1}^{M} N_k * F_k = 0`,
        where :math:`N_k` is the number of conformations sampled from state k.
        """
        return np.array(self._F_cov)

    @property
    def DeltaF(self) -> Float[np.ndarray, "M M"]:
        r"""Free energy difference between states.
        :math:`\mathrm{DeltaF}[i,j]` is the free energy difference between state j and state i,
        i.e., :math:`\mathrm{DeltaF}[i,j] = F[j] - F[i]` .
        """
        return np.array(self._DeltaF)

    @property
    def DeltaF_std(self) -> Float[np.ndarray, "M M"]:
        r"""Standard deviation of the free energy difference between states.
        :math:`\mathrm{DeltaF_std}[i,j]` is the standard deviation of the free energy
        difference :math:`\mathrm{DeltaF}[i,j]`.
        """
        return np.array(self._DeltaF_std)

    @property
    def log_prob_mix(self) -> Float[np.ndarray, "*R N"]:
        """the log probability density of conformations in the mixture distribution.
        With bootstrap, one row per bootstrap replicate."""
        return np.array(self._log_prob_mix)

    @jaxtyped(typechecker=beartype)
    def calculate_free_energies_of_perturbed_states(
        self, energy_perturbed: Real[HostOrDeviceArray, "L N"]
    ) -> dict[str, Float[np.ndarray, "..."]]:
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

        Raises:
            jaxtyping.TypeCheckError: (a subclass of TypeError) if energy_perturbed
                is not a 2-D real array.
            ValueError: if energy_perturbed does not have one column per conformation.
        """

        ## the type and number of dimensions are checked by jaxtyping; the number of
        ## columns has to match the energy matrix given to the constructor, which
        ## the annotation cannot see
        if energy_perturbed.shape[1] != self.N:
            raise ValueError(
                "the number of columns in energy_perturbed has to be equal to the number of columns in energy."
            )
        energy_perturbed = energy_perturbed.astype(np.float64)

        L = energy_perturbed.shape[0]

        if not self.energy_on_device:
            ## stream the perturbed energies like the energy matrix: transposed to
            ## (N, L) in host memory so that each batch is a contiguous block
            energy_perturbed_T = np.ascontiguousarray(energy_perturbed.T)

        F = None
        F_cov = None
        F_std = None

        if not self.bootstrap and self.batch_size is not None:
            ## same results as the unbatched branch below, accumulated over batches
            ## of conformations; the N x (M + L) matrix P is never formed
            if self.energy_on_device:
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
            else:
                F = _perturbed_F_streamed(
                    energy_perturbed_T, self._log_prob_mix, self.batch_size
                )
                gram = _extended_gram_streamed(
                    self._F,
                    F,
                    self._energy_T,
                    energy_perturbed_T,
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
            if self.batch_size is not None and self.energy_on_device:
                energy_perturbed = jax.device_put(energy_perturbed)
            for k in range(self.bootstrap_num_rep):
                if self.batch_size is None:
                    du = energy_perturbed[:, self._conf_idx[k]] + self._log_prob_mix[k]
                    F = -logsumexp(-du, axis=1)
                elif not self.energy_on_device:
                    F = _perturbed_F_streamed(
                        energy_perturbed_T,
                        self._log_prob_mix[k],
                        self.batch_size,
                        self._conf_idx[k],
                    )
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
    F: Float[Array, " M"],
    energy: Float[Array, "M N"],
    num_conf: Float[Array, " M"],
    batch_size: int,
    conf_idx: Int[Array, " N"] | None = None,
    with_gram: bool = True,
) -> tuple[Float[Array, " N"], Float[Array, "M M"]]:
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
def _perturbed_F_in_batches(
    energy_perturbed: Float[Array, "L N"],
    log_prob_mix: Float[Array, " N"],
    batch_size: int,
    conf_idx: Int[Array, " N"] | None = None,
) -> Float[Array, " L"]:
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
    F: Float[Array, " M"],
    F_perturbed: Float[Array, " L"],
    energy: Float[Array, "M N"],
    energy_perturbed: Float[Array, "L N"],
    log_prob_mix: Float[Array, " N"],
    batch_size: int,
) -> Float[Array, "M+L M+L"]:
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


#### Streaming from host memory; see _host_batches in utils ####
## In streaming mode the energy matrix is kept transposed as a host (N, M) array
## energy_T, and log_prob_mix is kept in host memory as well. Each function makes
## one pass over the conformations and blocks on every batch so that at most the
## current and the prefetched batch are on the device.


@partial(jit, static_argnames=["with_gram"])
def _add_gram_batch(
    ptp: Float[Array, "M M"],
    F: Float[Array, " M"],
    logn: Float[Array, " M"],
    u_T: Float[Array, "B M"],
    with_gram: bool,
) -> tuple[Float[Array, "M M"], Float[Array, " B"]]:
    lse, p = _mbar_weights(F, logn, u_T)
    if with_gram:
        ptp = ptp + p.T @ p
    return ptp, lse


def _log_prob_mix_and_gram_streamed(
    F: Float[Array, " M"],
    energy_T: Float[np.ndarray, "N M"],
    num_conf: Float[Array, " M"],
    batch_size: int,
    conf_idx: Int[np.ndarray, " N"] | None = None,
    with_gram: bool = True,
) -> tuple[Float[np.ndarray, " N"], Float[Array, "M M"]]:
    """Same as _log_prob_mix_and_gram_in_batches, streaming energy_T, or its rows
    conf_idx if given, to the device in batches. log_prob_mix is returned in host
    memory."""
    logn = jnp.log(num_conf)
    N, M = energy_T.shape
    ptp = jnp.zeros((M, M))
    lse_batches = []
    for u_T in _host_batches(_rows(energy_T, conf_idx), N, batch_size):
        ptp, lse = jax.block_until_ready(_add_gram_batch(ptp, F, logn, u_T, with_gram))
        lse_batches.append(np.asarray(lse))
    log_prob_mix = np.concatenate(lse_batches)
    gram = ptp / (num_conf[:, None] * num_conf[None, :])
    return log_prob_mix, gram


@jit
def _add_perturbed_lse_batch(
    acc: Float[Array, " L"], up_T: Float[Array, "B L"], lpm: Float[Array, " B"]
) -> Float[Array, " L"]:
    return jnp.logaddexp(acc, logsumexp(-(up_T + lpm[:, None]), axis=0))


def _perturbed_F_streamed(
    energy_perturbed_T: Float[np.ndarray, "N L"],
    log_prob_mix: Float[np.ndarray, " N"],
    batch_size: int,
    conf_idx: Int[np.ndarray, " N"] | None = None,
) -> Float[Array, " L"]:
    """Same as _perturbed_F_in_batches, streaming the transposed perturbed energies
    energy_perturbed_T (a host (N, L) array; its rows conf_idx if given) together
    with log_prob_mix (a host array aligned with the possibly resampled rows)."""
    N, L = energy_perturbed_T.shape
    rows = _rows(energy_perturbed_T, conf_idx)
    acc = jnp.full((L,), -jnp.inf)
    for up_T, lpm in _host_batches(
        lambda start, stop: (rows(start, stop), log_prob_mix[start:stop]),
        N,
        batch_size,
    ):
        acc = jax.block_until_ready(_add_perturbed_lse_batch(acc, up_T, lpm))
    return -acc


@jit
def _add_extended_gram_batch(
    acc: Float[Array, "M+L M+L"],
    F_ext: Float[Array, " M+L"],
    u_T: Float[Array, "B M"],
    up_T: Float[Array, "B L"],
    lpm: Float[Array, " B"],
) -> Float[Array, "M+L M+L"]:
    P = jnp.exp(F_ext - jnp.concatenate([u_T, up_T], axis=1) - lpm[:, None])
    return acc + P.T @ P


def _extended_gram_streamed(
    F: Float[Array, " M"],
    F_perturbed: Float[Array, " L"],
    energy_T: Float[np.ndarray, "N M"],
    energy_perturbed_T: Float[np.ndarray, "N L"],
    log_prob_mix: Float[np.ndarray, " N"],
    batch_size: int,
) -> Float[Array, "M+L M+L"]:
    """Same as _extended_gram_in_batches, streaming the transposed sampled and
    perturbed energies and log_prob_mix from host memory."""
    F_ext = jnp.concatenate([F, F_perturbed])
    K = F_ext.shape[0]
    acc = jnp.zeros((K, K))
    for u_T, up_T, lpm in _host_batches(
        lambda start, stop: (
            energy_T[start:stop],
            energy_perturbed_T[start:stop],
            log_prob_mix[start:stop],
        ),
        energy_T.shape[0],
        batch_size,
    ):
        acc = jax.block_until_ready(_add_extended_gram_batch(acc, F_ext, u_T, up_T, lpm))
    return acc


def _fits_on_device(nbytes: int, margin: float = 1.25) -> bool:
    """Whether an array of nbytes, plus margin for temporaries, fits in the free
    memory of the default device. Devices that do not report memory statistics
    (e.g. the CPU) are assumed to fit."""
    stats = jax.devices()[0].memory_stats()
    if not stats or "bytes_limit" not in stats:
        return True
    free = stats["bytes_limit"] - stats["bytes_in_use"]
    return nbytes * margin <= free


def _covariance_from_gram(
    gram: Float[Array, "K K"], weights: Float[Array, " K"], rtol: float
) -> Float[Array, "K K"]:
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


def _bootstrap_conf_idx(
    key: PRNGKeyArray,
    num_conf: Real[HostOrDeviceArray, " M"],
    bootstrap_block_size: int,
) -> Int[Array, " N"]:
    """Indices of a circular block-bootstrap resample of the conformations, drawn
    within each state's contiguous block of columns."""
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
