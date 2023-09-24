from functools import partial
import jax.numpy as jnp
from jax import grad, hessian, jit, value_and_grad
import numpy as np
import jax
from jax.config import config

config.update("jax_enable_x64", True)
from jax.scipy.special import logsumexp
from jax import random
import blackjax
from optax import sgd
import optax
from collections import namedtuple
from utils import fmin_newton

class BayesMBAR:
    def __init__(
        self,
        rng_key,
        energy,
        num_conf,
        prior,
        mean,
        kernel,
        state_cv,
        sample_size: int = 1000,
        warmup_steps: int = 500,
        optimize_steps: int = 10000,
    ) -> None:
        self.rng_key = rng_key

        self.energy = jnp.float64(energy)
        self.num_conf = jnp.int32(num_conf)

        self.prior = prior
        self.mean_name = mean
        self.kernel_name = kernel

        if state_cv is not None:
            self.state_cv = state_cv
            self._state_cv = self.state_cv[1:]

        self.sample_size = sample_size
        self.warmup_steps = warmup_steps
        self.optimize_steps = optimize_steps

        self.m = self.energy.shape[0]
        self.n = self.energy.shape[1]

        ## solve the mbar equation in terms of dF
        dF_init = jnp.zeros((self.m - 1,))
        f = jit(value_and_grad(_compute_loss_likelihood_of_dF))
        hess = jit(hessian(_compute_loss_likelihood_of_dF))
        res = fmin_newton(f, hess, dF_init, args=(self.energy, self.num_conf))
        dF = res["x"]
        self._dF_mode_ll = dF

        ## sample dF from the likelihood
        self.rng_key, subkey = random.split(self.rng_key)
        logdensity = lambda dF: _compute_log_likelihood_of_dF(
            dF, self.energy, self.num_conf
        )
        self._dF_samples_ll = _sample_dF_from_logdensity(
            subkey,
            self._dF_mode_ll,
            logdensity,
            self.warmup_steps,
            self.sample_size,
        )
        self._dF_mean_ll = jnp.mean(self._dF_samples_ll, axis=0)
        self._dF_cov_ll = jnp.cov(self._dF_samples_ll.T)
        self._dF_prec_ll = jnp.linalg.inv(self._dF_cov_ll)

        self._F_mode_ll = _dF_to_F(self._dF_mode_ll, self.num_conf)
        self._F_samples_ll = _dF_to_F(self._dF_samples_ll, self.num_conf)
        self._F_mean_ll = jnp.mean(self._F_samples_ll, axis=0)
        self._F_cov_ll = jnp.cov(self._F_samples_ll.T)

        print("sampling from the likelihood is done")

        if self.prior == "normal":
            _data = {
                "energy": self.energy,
                "num_conf": self.num_conf,
                "dF_mean_ll": self._dF_mean_ll,
                "dF_prec_ll": self._dF_prec_ll,
                "state_cv": self._state_cv,
            }

            ## mean function of the prior
            if self.mean_name == "constant":
                self.mean_order = 0
                self.mean = partial(_mean, order=self.mean_order)
            elif self.mean_name == "linear":
                self.mean_order = 1
                self.mean = partial(_mean, order=self.mean_order)
            elif self.mean_name == "quadratic":
                self.mean_order = 2
                self.mean = partial(_mean, order=self.mean_order)

            ## learn the hyperparameters of the prior
            if self.kernel_name == "SE":
                self.kernel = _kernel_SE
            elif self.kernel_name == "Matern52":
                self.kernel = _kernel_Matern52
            elif self.kernel_name == "Matern32":
                self.kernel = _kernel_Matern32
            elif self.kernel_name == "RQ":
                self.kernel = _kernel_RQ

            ## initialize the hyperparameters based on the mode of the likelihood
            params = _init_params(
                self.mean_order,
                self.kernel_name,
                self._dF_mode_ll,
                self._state_cv,
                self.num_conf,
            )
            raw_params = _params_to_raw(params)

            ## optimize the hyperparameters
            self.rng_key, subkey = random.split(self.rng_key)
            optimizer = sgd(learning_rate=1e-3, momentum=0.9, nesterov=True)
            opt_state = optimizer.init(raw_params)

            @partial(jit, static_argnames=["mean", "kernel"])
            def step(key, raw_params, opt_state, mean, kernel, data):
                loss, grads = _compute_elbo_loss(key, raw_params, mean, kernel, data)
                update, opt_state = optimizer.update(grads, opt_state)
                raw_params = optax.apply_updates(raw_params, update)
                return loss, raw_params, opt_state

            for i in range(optimize_steps):
                loss, raw_params, opt_state = step(
                    subkey, raw_params, opt_state, self.mean, self.kernel, _data
                )
                self.rng_key, subkey = random.split(self.rng_key)
                if i % 100 == 0:
                    params = _params_from_raw(raw_params)
                    print(f"step: {i:>10d}, loss: {loss:10.4f}", _print_params(params))

            self.params = _params_from_raw(raw_params)
            self._dF_mean_prior = self.mean(self.params["mean"], self._state_cv)
            self._dF_cov_prior = self.kernel(
                self.params["kernel"],
                self._state_cv,
            )
            self._dF_prec_prior = jnp.linalg.inv(self._dF_cov_prior)

            ## solve for the mode of the posterior
            f = jit(value_and_grad(_compute_loss_joint_likelihood_of_dF))
            hess = jit(hessian(_compute_loss_joint_likelihood_of_dF))
            res = fmin_newton(
                f,
                hess,
                self._dF_mode_ll,
                args=(
                    self.energy,
                    self.num_conf,
                    self._dF_mean_prior,
                    self._dF_prec_prior,
                ),
            )
            self._dF_mode_posterior = res["x"]

            ## sample dF from the posterior
            logdensity = lambda dF: _compute_log_joint_likelihood_of_dF(
                dF, self.energy, self.num_conf, self._dF_mean_prior, self._dF_prec_prior
            )
            self.rng_key, subkey = random.split(self.rng_key)
            self._dF_samples_posterior = _sample_dF_from_logdensity(
                subkey,
                self._dF_mode_posterior,
                logdensity,
                self.warmup_steps,
                self.sample_size,
            )
            self._dF_mean_posterior = jnp.mean(self._dF_samples_posterior, axis=0)
            self._dF_cov_posterior = jnp.cov(self._dF_samples_posterior.T)
            self._dF_prec_posterior = jnp.linalg.inv(self._dF_cov_posterior)

            self._F_mode_posterior = _dF_to_F(self._dF_mode_posterior, self.num_conf)
            self._F_samples_posterior = _dF_to_F(
                self._dF_samples_posterior, self.num_conf
            )
            self._F_mean_posterior = jnp.mean(self._F_samples_posterior, axis=0)
            self._F_cov_posterior = jnp.cov(self._F_samples_posterior.T)

    @property
    def F_mode(self):
        if self.prior == "uniform":
            F_mode = self._F_mode_ll
        elif self.prior == "normal":
            F_mode = self._F_mode_posterior
        return jax.device_put(F_mode, jax.devices("cpu")[0])

    @property
    def F_mean(self):
        if self.prior == "uniform":
            F_mean = self._F_mean_ll
        elif self.prior == "normal":
            F_mean = self._F_mean_posterior
        return jax.device_put(F_mean, jax.devices("cpu")[0])

    @property
    def F_cov(self):
        if self.prior == "uniform":
            F_cov = self._F_cov_ll
        elif self.prior == "normal":
            F_cov = self._F_cov_posterior
        F_cov = F_cov - jnp.diag(1.0 / self.num_conf) + 1.0 / self.num_conf.sum()

        ## if the diagnoal elements of F_cov are negetive, set them to 1e-4
        condition = jnp.eye(F_cov.shape[0], dtype=bool) & (F_cov <= 0)
        F_cov = jnp.where(condition, 1e-4, F_cov)
        return jax.device_put(F_cov, jax.devices("cpu")[0])

    @property
    def F_std(self):
        return jnp.sqrt(jnp.diag(self.F_cov))

    @property
    def F_samples(self):
        if self.prior == "uniform":
            F_samples = self._F_samples_ll
        elif self.prior == "normal":
            F_samples = self._F_samples_posterior
        return jax.device_put(F_samples, jax.devices("cpu")[0])

    @property
    def DeltaF_mode(self):
        return self.F_mode[None, :] - self.F_mode[:, None]

    @property
    def DeltaF_mean(self):
        return self.F_mean[None, :] - self.F_mean[:, None]

    @property
    def DeltaF_std(self):
        DeltaF_cov = (
            jnp.diag(self.F_cov)[:, None]
            + jnp.diag(self.F_cov)[None, :]
            - 2 * self.F_cov
        )
        return jnp.sqrt(DeltaF_cov)


def _dF_to_F(dF, num_conf):
    if dF.ndim == 1:
        F = jnp.concatenate([jnp.zeros((1,)), dF])
    elif dF.ndim == 2:
        F = jnp.concatenate([jnp.zeros((dF.shape[0], 1)), dF], axis=1)
    pi = num_conf / num_conf.sum()
    F = F - jnp.sum(pi * F, axis=-1, keepdims=True)
    return F


def _compute_loss_joint_likelihood_of_dF(dF, energy, num_conf, mean_prior, prec_prior):
    loss = -_compute_log_joint_likelihood_of_dF(
        dF, energy, num_conf, mean_prior, prec_prior
    )
    loss = loss / (energy.shape[1] + energy.shape[0] - 1)
    return loss


def _compute_log_joint_likelihood_of_dF(dF, energy, num_conf, mean_prior, prec_prior):
    logp = -0.5 * jnp.dot(dF - mean_prior, jnp.dot(prec_prior, dF - mean_prior))
    logp = logp + _compute_log_likelihood_of_dF(dF, energy, num_conf)
    return logp


@partial(value_and_grad, argnums=1)
def _compute_elbo_loss(rng_key, raw_params, mean, kernel, data):
    energy = data["energy"]
    num_conf = data["num_conf"]
    state_cv = data["state_cv"]
    dF_prec_ll = data["dF_prec_ll"]
    dF_mean_ll = data["dF_mean_ll"]

    params = _params_from_raw(raw_params)
    mean_prior = mean(params["mean"], state_cv)
    cov_prior = kernel(params["kernel"], state_cv)
    mu_prop, cov_prop = _compute_proposal_dist(
        mean_prior, cov_prior, dF_mean_ll, dF_prec_ll
    )
    dFs = random.multivariate_normal(rng_key, mu_prop, cov_prop, shape=(1024,))
    Fs = jnp.concatenate([jnp.zeros((dFs.shape[0], 1)), dFs], axis=1)
    elbo = jax.vmap(_compute_log_likelihood_of_F, in_axes=(0, None, None))(
        Fs, energy, num_conf
    )
    elbo = jnp.mean(elbo)
    elbo = elbo - _compute_kl_divergence(mu_prop, cov_prop, mean_prior, cov_prior)
    return -elbo


def _compute_kl_divergence(mu0, cov0, mu1, cov1):
    L0 = jnp.linalg.cholesky(cov0)
    L1 = jnp.linalg.cholesky(cov1)

    M = jax.scipy.linalg.solve_triangular(L1, L0, lower=True)
    y = jax.scipy.linalg.solve_triangular(L1, mu1 - mu0, lower=True)

    kl = 0.5 * (
        jnp.sum(M**2)
        + jnp.sum(y**2)
        - mu0.shape[0]
        + 2 * jnp.sum(jnp.log(jnp.diag(L1)) - jnp.log(jnp.diag(L0)))
    )
    return kl


def _compute_proposal_dist(mean_prior, cov_prior, dF_mean_ll, dF_prec_ll):
    prec_prior = jnp.linalg.inv(cov_prior)
    prec = dF_prec_ll + prec_prior
    cov = jnp.linalg.inv(prec)
    mu = jnp.dot(cov, jnp.dot(dF_prec_ll, dF_mean_ll) + jnp.dot(prec_prior, mean_prior))
    return mu, cov


def _print_params(params):
    res = "beta: "
    for i in range(params["mean"]["beta"].shape[0]):
        res += f'{params["mean"]["beta"][i].item():.4f}, '

    res += f'scale: {params["kernel"]["scale"].item():.4f}, '

    res += "l_scale: "
    for i in range(params["kernel"]["length_scale"].shape[0]):
        res += f'{params["kernel"]["length_scale"][i].item():.4f}, '

    if "alpha" in params["kernel"].keys():
        res += f'alpha: {params["kernel"]["alpha"].item():.4f}, '

    res += 'dscale: '
    for i in range(params["kernel"]["dscale"].shape[0]):
        res += f'{params["kernel"]["dscale"][i].item():.4f}, '
    return res


def _expand(x, order):
    xx = [jnp.ones((x.shape[0], 1))]
    for i in range(order):
        xx.append(x ** (i + 1))
    xx = jnp.concatenate(xx, axis=-1)
    return xx


def _mean(params, x, order):
    xx = _expand(x, order)
    return jnp.sum(params["beta"] * xx, axis=-1)


def _params_from_raw(raw_params):
    params = {}
    params["mean"] = raw_params["mean"]
    params["kernel"] = _kernel_params_from_raw(raw_params["kernel"])
    return params


def _params_to_raw(params):
    raw_params = {}
    raw_params["mean"] = params["mean"]
    raw_params["kernel"] = _kernel_params_to_raw(params["kernel"])
    return raw_params


def _init_mean_params(order, dF, state_cv):
    x = _expand(state_cv, order)
    beta = jnp.linalg.lstsq(x, dF, rcond=None)[0]
    params = {"beta": beta}
    return params


def _init_params(mean_order, kernel_name, dF, state_cv, num_conf):
    params = {}
    params["mean"] = _init_mean_params(mean_order, dF, state_cv)
    params["kernel"] = _init_kernel_params(kernel_name, dF, state_cv, num_conf)
    return params


def _init_kernel_params(kernel_name, dF, state_cv, num_conf):
    params = {}
    params["scale"] = jnp.std(dF)
    params["length_scale"] = (state_cv.max(0) - state_cv.min(0)) / state_cv.shape[0]
    params["dscale"] = jnp.ones_like(dF)*jnp.std(dF)

    if kernel_name == "RQ":
        params["alpha"] = jnp.ones((1,)) * 10

    return params


def _kernel_RQ(params, x):
    scale = params["scale"]
    length_scale = params["length_scale"]
    dscale = params["dscale"]
    alpha = params["alpha"]
    x = x / length_scale
    ds = _compute_squared_distance(x)
    return scale**2 * (1 + ds / (2 * alpha)) ** (-alpha) + dscale**2 * jnp.eye(
        ds.shape[0]
    )


def _kernel_SE(params, x):
    scale = params["scale"]
    length_scale = params["length_scale"]
    dscale = params["dscale"]
    x = x / length_scale
    ds = _compute_squared_distance(x)
    return scale**2 * jnp.exp(-0.5 * ds) + dscale**2 * jnp.eye(ds.shape[0])


def _kernel_Matern52(params, x):
    scale = params["scale"]
    length_scale = params["length_scale"]
    dscale = params["dscale"]
    x = x / length_scale
    ds = _compute_squared_distance(x)
    d = jnp.sqrt(ds + 1e-18)
    return scale**2 * (1 + jnp.sqrt(5.0) * d + 5.0 / 3.0 * ds) * jnp.exp(
        -jnp.sqrt(5.0) * d
    ) + dscale**2 * jnp.eye(ds.shape[0])


def _kernel_Matern32(params, x):
    scale = params["scale"]
    length_scale = params["length_scale"]
    dscale = params["dscale"]
    x = x / length_scale
    ds = _compute_squared_distance(x)
    d = jnp.sqrt(ds + 1e-18)
    return scale**2 * (1 + jnp.sqrt(3.0) * d) * jnp.exp(
        -jnp.sqrt(3.0) * d
    ) + dscale**2 * jnp.eye(ds.shape[0])


def _kernel_params_from_raw(raw_params):
    params = {}
    params["scale"] = jax.nn.softplus(raw_params["raw_scale"])
    params["length_scale"] = jax.nn.softplus(raw_params["raw_length_scale"])
    params["dscale"] = jax.nn.softplus(raw_params["raw_dscale"])
    if "raw_alpha" in raw_params.keys():
        params["alpha"] = jax.nn.softplus(raw_params["raw_alpha"])
    return params


def _kernel_params_to_raw(params):
    raw_params = {}
    raw_params["raw_scale"] = jnp.log(jnp.exp(params["scale"]) - 1)
    raw_params["raw_length_scale"] = jnp.log(jnp.exp(params["length_scale"]) - 1)
    raw_params["raw_dscale"] = jnp.log(jnp.exp(params["dscale"]) - 1)
    if "alpha" in params.keys():
        raw_params["raw_alpha"] = jnp.log(jnp.exp(params["alpha"]) - 1)
    return raw_params


def _compute_squared_distance(x):
    x1 = x[:, None, :]
    x2 = x[None, :, :]
    return jnp.sum((x1 - x2) ** 2, axis=-1)


def _sample_dF_from_logdensity(rng_key, init_dF, logdensity, warmup_steps, num_samples):
    ## warmup to find step size and mass matrix
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity,
        is_mass_matrix_diagonal=False,
        progress_bar=True,
    )
    rng_key, subkey = random.split(rng_key)
    (state, parameters), _ = warmup.run(subkey, init_dF, num_steps=warmup_steps)

    ## sample using nuts
    rng_key, subkey = random.split(rng_key)
    kernel = blackjax.nuts(logdensity, **parameters).step
    states = _sample_loop(subkey, kernel, state, num_samples)
    return states.position


def _sample_loop(rng_key, kernel, init_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, init_state, keys)
    return states


def _compute_log_likelihood_of_F(F, energy, num_conf):
    logn = jnp.log(num_conf)
    u = energy.T - F - logn
    l = jnp.sum(num_conf * F) - logsumexp(-u, axis=1).sum()
    return l


def _compute_log_likelihood_of_dF(dF, energy, num_conf):
    F = jnp.concatenate([jnp.zeros((1,)), dF])
    return _compute_log_likelihood_of_F(F, energy, num_conf)


def _compute_loss_likelihood_of_dF(dF, energy, num_conf):
    return -_compute_log_likelihood_of_dF(dF, energy, num_conf) / num_conf.sum()


