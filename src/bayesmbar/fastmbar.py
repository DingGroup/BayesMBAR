from copy import deepcopy
from time import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp
import numpy.typing as npt
from .utils import _solve_mbar

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
    ) -> None:
        """Initializer for the class FastMBAR

        Parameters
        ----------
        energy : 2D ndarray
            It has a size of (M x N), where M is the number of states
            and N is the total number of conformations. The entry energy[i,j]
            is the reduced (unitless) energy of conformation j in state i.
            If bootstrapping is used to calculate the uncertainty, the order
            of conformations matters. Conformations sampled from one state
            need to occpy a continuous chunk of collumns. Conformations sampled
            from state k need to occupy collumns to the left of conformations
            sampled from state l if k < l. If bootstrapping is not used, then
            the order of conformation does not matter.
        num_conf: 1D int ndarray
            It should have a size of M, where num_conf[i] is the num of
            conformations sampled from state i. Therefore, np.sum(num_conf)
            has to be equal to N. All entries in num_conf have to be strictly
            greater than 0.
        bootstrap: bool, optional (default=False)
            If bootstrap is True, the uncertainty of the calculated free energies
            will be estimate using block bootstraping.
        bootstrap_block_size: int, optional (default=3)
            block size used in block bootstrapping
        bootstrap_num_rep: int, optional (default=100)
            number of repreats in block bootstrapping
        verbose: bool, optional (default=False)
            if verbose is true, the detailed information of solving MBAR equations
            is printed.
        method: str, optional (default="Newton")
            the method used to solve the MBAR equation. The default is Newton's method.
        """

        #### check the parameters: energy and num_conf
        ## Note that both self.energy and self.conf will be on CPUs no matter
        ## if cuda is True or False.

        ## check energy
        if isinstance(energy, np.ndarray):
            if energy.ndim != 2:
                raise ValueError("energy has to be two dimensional")
            self.energy = energy.astype(np.float64)
        else:
            raise TypeError("energy has to be a 2-D ndarray")

        ## check num_conf
        if isinstance(num_conf, np.ndarray):
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
        if method not in ["Newton", "L-BFGS-B"]:
            raise ValueError("method has to be Newton or L-BFGS-B.")
        self.method = method

        # ## solve the MBAR equation
        if self.bootstrap is False:
            dF_init = jnp.zeros(self.M - 1)
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
            self._conf_idx = []
            for _ in range(self.bootstrap_num_rep):
                conf_idx = _bootstrap_conf_idx(num_conf, self.bootstrap_block_size)
                dF = _solve_mbar(
                    dF_init,
                    self.energy[:, conf_idx],
                    self.num_conf,
                    self.method,
                    verbose=self.verbose,
                )
                dF_init = dF
                dFs.append(deepcopy(dF))

                F = jnp.concat([jnp.zeros(1), dF])
                ## shift F such that \sum_i F[i]*num_conf[i] = 0
                pi = self.num_conf / self.N
                F = F - jnp.sum(pi * F)

                b = -F - jnp.log(self.num_conf / self.N)
                log_prob_mix.append(
                    logsumexp(-(self.energy[:, conf_idx] + b[:, None]), axis=0)
                )
                self._conf_idx.append(conf_idx)

            dF = jnp.stack(dFs, axis=1)
            F = jnp.concat([jnp.zeros((1, dF.shape[1])), dF], axis=0)

            ## shift F such that \sum_i F[i]*num_conf[i] = 0
            pi = self.num_conf / self.N
            self._F_bootstrap = F - jnp.sum(pi[:, None] * F, axis=0)

            self._F = jnp.mean(self._F_bootstrap, axis=1)
            self._F_std = jnp.std(self._F_bootstrap, axis=1)
            self._F_cov = jnp.cov(self._F_bootstrap)

            self._log_prob_mix = jnp.stack(log_prob_mix, axis=0)
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

        Parameters
        -----------
        energy_perturbed: 2-D float ndarray with size of (L,N)
            each row of the energy_perturbed matrix represents a state and
            the value energy_perturbed[l,n] represents the reduced energy
            of the n'th conformation in the l'th perturbed state.

        Returns
        -------
        results: dict
            a dictionary containing the following keys:

            **F** - the free energy of the perturbed states.

            **F_std** - the standard deviation of the free energy of the perturbed states.

            **F_cov** - the covariance between the free energies of the perturbed states.

            **DeltaF** - :math:`\mathrm{DeltaF}[k,l]` is the free energy difference between state
            :math:`k` and state :math:`l`, i.e., :math:`\mathrm{DeltaF}[k,l] = F[l] - F[k]` .

            **DeltaF_std** - the standard deviation of the free energy difference.
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

        F = None
        F_cov = None
        F_std = None

        if not self.bootstrap:
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
            for k in range(self.bootstrap_num_rep):
                du = energy_perturbed[:, self._conf_idx[k]] + self._log_prob_mix[k]
                F = -logsumexp(-du, axis=1)
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





def _bootstrap_conf_idx(num_conf, bootstrap_block_size):
    num_conf_cumsum = jnp.cumsum(num_conf, axis=0).tolist()
    num_conf_cumsum.pop(-1)
    num_conf_cumsum = [0] + num_conf_cumsum
    conf_idx = []

    key = jr.PRNGKey(int(time() * 1000))
    for i in range(len(num_conf)):
        len_seq = int(num_conf[i])  ## using circular block bootstrap
        num_sample_block = int(jnp.ceil(len_seq / bootstrap_block_size))
        key, subkey = jr.split(key)
        idxs = jr.randint(subkey, (num_sample_block,), 0, len_seq)
        sample_idx = jnp.concat(
            [jnp.arange(idx, idx + bootstrap_block_size) for idx in idxs]
        )
        sample_idx = jnp.remainder(sample_idx, len_seq)
        sample_idx = sample_idx[0:len_seq]
        sample_idx = sample_idx + num_conf_cumsum[i]
        conf_idx.append(sample_idx)
    conf_idx = jnp.concat(conf_idx)
    return conf_idx
