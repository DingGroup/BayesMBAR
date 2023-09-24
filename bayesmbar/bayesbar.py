import math
import random
import numpy as np
from numpy import ndarray
import scipy.optimize as optimize
import scipy.integrate as integrate
import jax
from jax.config import config

config.update("jax_enable_x64", True)
from jax import grad, hessian, jit, value_and_grad
import jax.numpy as jnp
from utils import fmin_newton
from tqdm import tqdm


class BayesBAR:
    """Bayesian Bennett acceptance ratio method"""

    def __init__(
        self,
        energy: ndarray,
        num_conf: ndarray,
        verbose=False,
        sample_size: int = 0,
    ):
        """Initializer of the BayesBAR class

        Args:
          energy: energy matrix in reduced units. Its size should be 2xN, where N is
              the total number of samples from the two states.
          num_conf: number of configurations in each state. Its size should be (2,).
          sample_size: the number of samples from the posterior distribution.
          use_gpu: whether to use gpus.
          verbose: whether to print running information
          method: Newton or L-BFGS-B
        """

        assert (
            energy.shape[0] == 2
        ), f"The energy matrix has {energy.shape[0]} rows. It has to have 2 rows."

        assert (
            len(num_conf) == 2
        ), f"The size of num_conf is {len(num_conf)}. It has to be 2."

        assert (
            energy.shape[1] == num_conf[0] + num_conf[1]
        ), "the number of column in energy is not equal to the sum of num_conf"

        self.energy = jnp.float64(energy)
        self.num_conf = jnp.int32(num_conf)

        self.n_0, self.n_1 = self.num_conf
        self.n = jnp.sum(self.num_conf)

        ## find the posterior mode which corresponds to the BAR solution
        dF_init = jnp.zeros(1, dtype=jnp.float64)
        dF_init = jnp.mean(self.energy[1] - self.energy[0]).reshape((-1,))
        
        f = jit(value_and_grad(_compute_loss))
        hess = jit(hessian(_compute_loss))
        res = fmin_newton(f, hess, dF_init, args=(self.energy, self.num_conf))
        self.dF_mode = res["x"]

        # self._compute_mode(method=self.method)
        ## compute posterior mean and standard deviation using numerical integration
        self.dF_mean, self.dF_std = _compute_posterior_mean_and_std(
            self.dF_mode, self.energy, self.num_conf
        )

        ## sampling from the posterior distribution
        self.sample_size = sample_size
        if self.sample_size > 0:
            self.dF_samples = _sample_from_posterior(
                self.dF_mode,
                self.dF_std,
                self.energy,
                self.num_conf,
                self.sample_size,
            )

        ## compute asymptotic standard deviation
        H = hessian(_compute_logp)(self.dF_mode, self.energy, self.num_conf)
        _dF_var_asymptotic = -1.0 / H - 1.0 / self.n_0 - 1.0 / self.n_1
        self._dF_std_asymptotic = jnp.reshape(jnp.sqrt(_dF_var_asymptotic), ())

        ## Bennett's uncertainty
        du = (
            self.energy[1, :]
            - self.energy[0, :]
            - jnp.log(self.n_1 / self.n_0)
            - self.dF_mode
        )
        f0 = jax.nn.sigmoid(-du[0 : self.n_0])
        f1 = jax.nn.sigmoid(du[self.n_0 :])
        _dF_var_bennett = (
            jnp.mean(f0**2) / (self.n_0 * jnp.mean(f0) ** 2)
            + jnp.mean(f1**2) / (self.n_1 * jnp.mean(f1) ** 2)
            - 1.0 / self.n_0
            - 1.0 / self.n_1
        )
        self._dF_std_bennett = jnp.sqrt(_dF_var_bennett)


@jit
def _compute_logp(dF, energy, num_conf):
    n_0, n_1 = num_conf
    du = energy[1, :] - energy[0, :] - jnp.log(n_1 / n_0)
    logp = n_1 * dF - jnp.logaddexp(jnp.zeros(1), dF - du).sum()
    return jnp.reshape(logp, ())


def _compute_loss(dF, energy, num_conf):
    logp = _compute_logp(dF, energy, num_conf)
    loss = -logp / num_conf.sum()
    return loss


def _compute_posterior(dF, energy, num_conf, dF_mode):
    logp_max = _compute_logp(dF_mode, energy, num_conf)

    ## The prior is chosen to be the uniform distribution over R,
    ## so the posterior is equal to the likelihood. The likelihood
    ## is shifted down by a constant which is the self.logp_max

    return jnp.exp(_compute_logp(dF, energy, num_conf) - logp_max)


def _compute_posterior_mean_and_std(dF_mode, energy, num_conf):
    ## compute the normalization constant Z
    f = lambda dF: _compute_posterior(dF, energy, num_conf, dF_mode)
    Z, Z_err = integrate.quad(jit(f), -np.inf, np.inf)

    ## posterior mean
    f = lambda dF: dF * _compute_posterior(dF, energy, num_conf, dF_mode)
    dF, dF_err = integrate.quad(jit(f), -np.inf, np.inf)
    dF_mean = dF / Z

    ## posterior standard deviation
    f = lambda dF: (dF - dF_mean) ** 2 * _compute_posterior(
        dF, energy, num_conf, dF_mode
    )
    dF_var, dF_var_err = integrate.quad(jit(f), -np.inf, np.inf)
    dF_var = dF_var / Z
    dF_std = math.sqrt(dF_var)
    return dF_mean, dF_std


def _sample_from_posterior(dF_mode, dF_std, energy, num_conf, size):
    """sample from the posterior using slice sampling
    (https://www.jstor.org/stable/3448413 )
    """

    ## dF_std is used as the estimate of the typical size of a slice.
    width = dF_std

    ## the size of a slice will be limited to max_size*width
    max_size = 3

    ## start from the posterior mode
    x0 = dF_mode
    samples = [x0]

    for _ in tqdm(range(size - 1)):
        ## sample the auxiliarxy random variable
        logp = _compute_logp(x0, energy, num_conf)
        z = logp - random.expovariate(1.0)

        ## find the slice interval using the "stepping out" procedure
        u = random.uniform(0.0, 1.0)
        L = x0 - u * width
        R = L + width

        v = random.uniform(0.0, 1.0)
        J = math.floor(v * max_size)
        K = max_size - 1 - J

        while J > 0 and z < _compute_logp(L, energy, num_conf):
            L = L - width
            J = J - 1

        while K > 0 and z < _compute_logp(R, energy, num_conf):
            R = R + width
            K = K - 1

        ## sampling from the interval using the "shrinkage" procedure
        while True:
            x1 = random.uniform(L, R)
            if z < _compute_logp(x1, energy, num_conf):
                samples.append(x1)
                x0 = x1
                break
            if x1 < x0:
                L = x1
            else:
                R = x1

    return jnp.array(samples).reshape(-1)
