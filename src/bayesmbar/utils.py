from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import hessian, jit, value_and_grad
from jax.scipy.special import logsumexp
from scipy import optimize


def _solve_mbar(
    dF_init, energy, num_conf, method, verbose, batch_size=None, conf_idx=None
):
    """Solve the MBAR equations for dF.

    If batch_size is None, the loss, gradient and Hessian are computed over the whole
    energy matrix at once. Otherwise they are accumulated over batches of
    batch_size configurations, so that the temporary memory scales with batch_size
    instead of with the number of configurations. In batched mode, conf_idx can
    select the columns of energy to use (e.g. a bootstrap resample) without
    forming energy[:, conf_idx].
    """
    if batch_size is None:
        if conf_idx is not None:
            raise ValueError("conf_idx is only supported with batch_size.")
        loss = _compute_loss_likelihood_of_dF
        value_and_grad_fn = _value_and_grad_loss_likelihood_of_dF
        hessian_fn = _hessian_loss_likelihood_of_dF
        args = (energy, num_conf)
    else:
        ## put the energy matrix on the device once instead of in every call;
        ## jax.device_put avoids the transient second copy made by jnp.asarray
        energy = jax.device_put(energy)
        num_conf = jnp.asarray(num_conf, dtype=energy.dtype)
        loss, value_and_grad_fn, hessian_fn = _batched_loss_functions(batch_size)
        args = (energy, num_conf, conf_idx)

    if method == "Newton":
        res = fmin_newton(
            value_and_grad_fn,
            hessian_fn,
            dF_init,
            args=args,
            verbose=verbose,
        )
        dF = res["x"]
    elif method == "L-BFGS":
        res = fmin_lbfgs(
            loss,
            dF_init,
            args=args,
            verbose=verbose,
        )
        dF = res["x"]

        # options = {"disp": verbose, "gtol": 1e-8}
        # f = jit(value_and_grad(_compute_loss_likelihood_of_dF))
        # results = optimize.minimize(
        #     lambda x: [np.array(r) for r in f(x, energy, num_conf)],
        #     dF_init,
        #     jac=True,
        #     method="L-BFGS-B",
        #     tol=1e-12,
        #     options=options,
        # )
        # dF = results["x"]
    return dF


def _compute_log_likelihood_of_F(F, energy, num_conf):
    """
    Compute the log likelihood of F.

    See Eq. (5) in the reference paper.

    Arguments:
        F (jnp.ndarray): Free energies of the states
        energy (jnp.ndarray): Energy matrix
        num_conf (jnp.ndarray): Number of configurations in each state

    Returns:
        jnp.ndarray: Log likelihood of F

    """

    logn = jnp.log(num_conf)
    u = energy.T - F - logn
    L = jnp.sum(num_conf * F) - logsumexp(-u, axis=1).sum()
    return L


def _compute_log_likelihood_of_dF(dF, energy, num_conf):
    """
    Compute the log likelihood of dF.

    Because F can only be determined up to an additive constant, we use dF instead of F as the parameter in both optimization and sampling.
    dF is defined as dF = [F_1 - F_0, F_2 - F_0, ..., F_m - F_0].

    See the doc of _compute_log_likelihood_of_F for more details on the arguments and the return value.
    """

    F = jnp.concatenate([jnp.zeros((1,)), dF])
    return _compute_log_likelihood_of_F(F, energy, num_conf)


def _compute_loss_likelihood_of_dF(dF, energy, num_conf):
    """
    Compute the loss function of dF based on the likelihood.

    The log likelihood of dF scales with the number of configurations. To make the loss function semi-invariant to the number of configurations, we divide the log likelihood by the total number of configurations. This helps to set a single tolerance for the optimization algorithm.

    Arguments:
        dF (jnp.ndarray): Free energy differences
        energy (jnp.ndarray): Energy matrix
        num_conf (jnp.ndarray): Number of configurations in each state

    Returns:
        jnp.ndarray: Loss function of dF
    """

    return -_compute_log_likelihood_of_dF(dF, energy, num_conf) / num_conf.sum()


## Built once at module level so that repeated solves with the same array shapes
## (e.g., bootstrap replicates) hit JAX's compilation cache.
_value_and_grad_loss_likelihood_of_dF = jit(
    value_and_grad(_compute_loss_likelihood_of_dF)
)
_hessian_loss_likelihood_of_dF = jit(hessian(_compute_loss_likelihood_of_dF))


def _loop_over_batches(body, n, batch_size, init):
    """Run carry = body(start, size, carry) over consecutive batches covering range(n).

    Full batches run in a fori_loop so that only one batch of intermediates is alive
    at a time; the last n % batch_size items are processed once at the end. size is
    always a Python int, so it can be used as a static slice size.
    """
    ## the loop body is traced even when it runs zero times, so its slices must fit
    batch_size = min(batch_size, n)
    num_full_batches = n // batch_size
    carry = jax.lax.fori_loop(
        0,
        num_full_batches,
        lambda i, c: body(i * batch_size, batch_size, c),
        init,
    )
    if n % batch_size:
        carry = body(num_full_batches * batch_size, n % batch_size, carry)
    return carry


def _energy_columns(energy, conf_idx, start, size):
    """Columns start:start+size of energy, or of energy[:, conf_idx] if conf_idx is
    given (e.g. a bootstrap resample), without forming energy[:, conf_idx]."""
    if conf_idx is None:
        return jax.lax.dynamic_slice_in_dim(energy, start, size, axis=1)
    idx = jax.lax.dynamic_slice_in_dim(conf_idx, start, size)
    return jnp.take(energy, idx, axis=1)


def _num_columns(energy, conf_idx):
    return energy.shape[1] if conf_idx is None else conf_idx.shape[0]


def _loss_grad_hess_in_batches(
    dF, energy, num_conf, conf_idx, batch_size, with_hessian
):
    """Loss, gradient and (optionally) Hessian of _compute_loss_likelihood_of_dF,
    accumulated over batches of configurations. If conf_idx is given, the energy
    matrix is energy[:, conf_idx].

    With p[n, k] = softmax_k(F_k + log N_k - u_k(x_n)), the negative log-likelihood
    and its derivatives with respect to F are sums over configurations:

        -L       = sum_n logsumexp_k(F_k + log N_k - u_k(x_n)) - sum_k N_k F_k
        d(-L)/dF = sum_n p[n] - N_k
        H(-L)    = diag(sum_n p[n]) - sum_n p[n]^T p[n]

    Derivatives with respect to dF drop the first state because F_0 = 0. Everything
    is divided by the total number of configurations as in
    _compute_loss_likelihood_of_dF.
    """
    F = jnp.concatenate([jnp.zeros((1,), dtype=dF.dtype), dF])
    logn = jnp.log(num_conf)
    M = energy.shape[0]

    def body(start, size, acc):
        u = _energy_columns(energy, conf_idx, start, size)
        return _add_mbar_sums(acc, F, logn, u.T, with_hessian)

    init = (jnp.zeros((), energy.dtype), jnp.zeros((M,), energy.dtype))
    if with_hessian:
        init = init + (jnp.zeros((M, M), energy.dtype),)
    sums = _loop_over_batches(
        body, _num_columns(energy, conf_idx), batch_size, init
    )
    return _loss_grad_hess_from_sums(sums, F, num_conf)


def _mbar_weights(F, logn, u_T):
    """For a batch of configurations with energies u_T of shape (B, M), return
    lse[n] = logsumexp_k(F_k + log N_k - u_k(x_n)) and the MBAR weights
    p[n, k] = exp(F_k + log N_k - u_k(x_n) - lse[n])."""
    logits = F + logn - u_T
    lse = logsumexp(logits, axis=1)
    return lse, jnp.exp(logits - lse[:, None])


def _add_mbar_sums(sums, F, logn, u_T, with_hessian=True):
    """Add one batch's contribution to (sum_n lse[n], sum_n p[n], sum_n p[n]^T p[n]);
    the last term only if with_hessian."""
    lse, p = _mbar_weights(F, logn, u_T)
    s = (lse.sum(), p.sum(axis=0))
    if with_hessian:
        s = s + (p.T @ p,)
    return jax.tree.map(jnp.add, sums, s)


def _loss_grad_hess_from_sums(sums, F, num_conf):
    """Loss, gradient and, if sums has the p^T p term, Hessian with respect to dF;
    see _loss_grad_hess_in_batches for the formulas."""
    N = num_conf.sum()
    loss = (sums[0] - jnp.sum(num_conf * F)) / N
    grad = (sums[1] - num_conf)[1:] / N
    if len(sums) == 2:
        return loss, grad
    hess = (jnp.diag(sums[1]) - sums[2])[1:, 1:] / N
    return loss, grad, hess


#### Streaming from host memory ####
## When the energy matrix does not fit on the device, it stays in host memory as a
## contiguous (N, M) array and every pass over the configurations copies one batch
## of rows at a time to the device.


def _host_batches(get_batch, n, batch_size):
    """Yield get_batch(start, stop) on the device for consecutive batches covering
    range(n). get_batch returns a host array or a tuple of host arrays.

    The copy of the next batch is issued (jax.device_put is asynchronous) before the
    current batch is yielded, so it overlaps with the computation on the current
    batch. Consumers should block on their result for each batch; otherwise the
    Python loop runs ahead and queues copies of many batches on the device at once.
    """
    next_batch = jax.device_put(get_batch(0, min(batch_size, n)))
    for start in range(0, n, batch_size):
        batch = next_batch
        stop = start + batch_size
        if stop < n:
            next_batch = jax.device_put(get_batch(stop, min(stop + batch_size, n)))
        yield batch


def _rows(array, conf_idx=None):
    """A get_batch for _host_batches returning rows start:stop of the host array,
    or of array[conf_idx] (e.g. a bootstrap resample) without forming it."""
    if conf_idx is None:
        return lambda start, stop: array[start:stop]
    return lambda start, stop: array[conf_idx[start:stop]]


_add_mbar_sums_jit = jit(_add_mbar_sums, static_argnames=["with_hessian"])


def _loss_grad_hess_streamed(
    dF, energy_T, num_conf, batch_size, conf_idx=None, with_hessian=True
):
    """Loss, gradient and (if with_hessian) Hessian of _compute_loss_likelihood_of_dF
    in one pass over the host (N, M) array energy_T, i.e. the transposed energy
    matrix, or over its rows conf_idx if given."""
    F = jnp.concatenate([jnp.zeros((1,), dtype=dF.dtype), dF])
    logn = jnp.log(num_conf)
    N, M = energy_T.shape
    sums = (jnp.zeros(()), jnp.zeros((M,)))
    if with_hessian:
        sums = sums + (jnp.zeros((M, M)),)
    for u_T in _host_batches(_rows(energy_T, conf_idx), N, batch_size):
        ## block so that at most the current and the prefetched batch are on the
        ## device; see _host_batches
        sums = jax.block_until_ready(
            _add_mbar_sums_jit(sums, F, logn, u_T, with_hessian)
        )
    return _loss_grad_hess_from_sums(sums, F, num_conf)


def _loss_grad_streamed(dF, energy_T, num_conf, batch_size, conf_idx=None):
    """Loss and gradient only; skips the O(M^2) per-conformation Hessian term."""
    return _loss_grad_hess_streamed(
        dF, energy_T, num_conf, batch_size, conf_idx, with_hessian=False
    )


def _solve_mbar_streamed(
    dF_init, energy_T, num_conf, verbose, batch_size, conf_idx=None, method="Newton"
):
    """Solve the MBAR equations, streaming the transposed energy matrix energy_T (a
    host (N, M) array), or its rows conf_idx (a host array of indices), to the
    device in batches.

    With method="Newton", each step usually costs one pass over the data that also
    accumulates the Hessian (see fmin_newton_fused). With method="L-BFGS", each
    function evaluation is a cheaper pass without the Hessian, but more passes are
    needed (see fmin_lbfgs_scipy).
    """
    num_conf = jnp.asarray(num_conf)
    args = (energy_T, num_conf, batch_size, conf_idx)
    if method == "Newton":
        res = fmin_newton_fused(
            _loss_grad_hess_streamed, dF_init, args=args, verbose=verbose
        )
    elif method == "L-BFGS":
        res = fmin_lbfgs_scipy(_loss_grad_streamed, dF_init, args=args, verbose=verbose)
    else:
        raise ValueError(f"Unknown method {method}.")
    return res["x"]


def fmin_lbfgs_scipy(fg, x_init, args=(), verbose=True, tol=1e-8, max_iter=500):
    """Minimize with SciPy's L-BFGS-B, where fg(x, *args) -> (value, grad).

    Unlike fmin_lbfgs, whose optax loop is compiled and runs entirely on the
    device, the loop here runs in Python, so fg can be a pass over data streamed
    from the host. tol is applied to the largest absolute component of the
    gradient (SciPy's gtol); ftol is set to 0 so that SciPy does not stop early on
    a small relative decrease of the loss before the gradient is small.
    """
    def fun(x):
        value, grad = fg(jnp.asarray(x), *args)
        return float(value), np.asarray(grad, dtype=np.float64)

    if verbose:
        print("==================================================================")
        print("                RUNNING THE L-BFGS-B METHOD (SCIPY)           \n")
        print("                           * * *                              \n")
        print(f"                    Tolerance = {tol:.3E}                    \n")

    res = optimize.minimize(
        fun,
        np.asarray(x_init, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"gtol": tol, "ftol": 0.0, "maxiter": max_iter},
    )

    if verbose:
        print(
            f"N_iter = {res.nit}, N_func = {res.nfev}, F = {res.fun:.12f}, "
            f"max |grad| = {np.max(np.abs(res.jac)):.3E}"
        )
        print(f"CONVERGENCE: {res.message}")
        print("==================================================================")

    return {
        "x": jnp.asarray(res.x),
        "N_iter": res.nit,
        "N_func": res.nfev,
        "F": res.fun,
        "success": res.success,
        "message": res.message,
    }


@lru_cache(maxsize=None)
def _batched_loss_functions(batch_size):
    """Batched counterparts of _compute_loss_likelihood_of_dF,
    _value_and_grad_loss_likelihood_of_dF and _hessian_loss_likelihood_of_dF.

    They are cached per batch_size so that repeated solves reuse the same function
    objects and hence the same compiled code.

    The loss has a custom VJP that returns the gradient from the same batched pass.
    Differentiating through the batch loop with autodiff would keep the
    intermediates of every batch for the backward pass, which defeats batching.
    """

    @jax.custom_vjp
    def loss(dF, energy, num_conf, conf_idx=None):
        return _loss_grad_hess_in_batches(
            dF, energy, num_conf, conf_idx, batch_size, False
        )[0]

    def loss_fwd(dF, energy, num_conf, conf_idx=None):
        value, grad = _loss_grad_hess_in_batches(
            dF, energy, num_conf, conf_idx, batch_size, False
        )
        return value, grad

    def loss_bwd(grad, g):
        ## only dF is differentiated; None means a zero cotangent for the data
        return (g * grad, None, None, None)

    loss.defvjp(loss_fwd, loss_bwd)

    @jit
    def value_and_grad_fn(dF, energy, num_conf, conf_idx=None):
        return _loss_grad_hess_in_batches(
            dF, energy, num_conf, conf_idx, batch_size, False
        )

    @jit
    def hessian_fn(dF, energy, num_conf, conf_idx=None):
        return _loss_grad_hess_in_batches(
            dF, energy, num_conf, conf_idx, batch_size, True
        )[2]

    return loss, value_and_grad_fn, hessian_fn


def fmin_newton(f, hess, x_init, args=(), verbose=True, eps=1e-10, max_iter=500):
    """Minimize a function with the Newton's method.

    For details of the Newton's method, see Chapter 9.5 of Prof. Boyd's book
    `Convex Optimization <https://web.stanford.edu/~boyd/cvxbook/>`_.

    Args:
        f (callable): the objective function to be minimized.
            f(x:Tensor, *args) ->  (y:Tensor, grad:Tensor), where
            y is the function value and grad is the gradient.
        hess (callable): the function to compute the Hessian matrix.
            hess(x:Tensor, *args) -> a two dimensional tensor.
        x_init (Tensor): the initial value.
        args (tuple): extra parameters for f and hess.
        verbose (bool): whether to print minimizing log information.
        eps (float): tolerance for stopping
        max_iter (int): maximum number of iterations.
    """

    ## constants used in backtracking line search
    alpha, beta = 0.01, 0.2

    ## Newton's method for minimizing the function
    indx_iter = 0

    N_func = 0

    if verbose:
        print("==================================================================")

        print("                RUNNING THE NEWTON'S METHOD                     \n")
        print("                           * * *                                \n")
        print(f"                    Tolerance EPS = {eps:.5E}                  \n")

    x = x_init

    while indx_iter < max_iter:
        loss, grad = f(x, *args)
        N_func += 1

        H = hess(x, *args)
        H = H + jnp.eye(H.shape[0]) * 1e-16

        newton_direction = jnp.linalg.solve(H, -grad)
        newton_decrement_square = jnp.sum(-grad * newton_direction)

        if verbose:
            print(
                f"At iterate {indx_iter:4d}; f= {loss.item():.5E};",
                f"|1/2*Newton_decrement^2|: {newton_decrement_square.item() / 2:.5E}\n",
            )

        if newton_decrement_square / 2.0 <= eps:
            break

        ## backtracking line search
        max_ls_iter = 100
        step_size = 1.0

        indx_ls_iter = 0
        while indx_ls_iter < max_ls_iter:
            target_loss, _ = f(x + step_size * newton_direction, *args)

            N_func += 1
            approximate_loss = loss + step_size * alpha * (-newton_decrement_square)
            if target_loss < approximate_loss:
                break
            else:
                step_size = step_size * beta
            indx_ls_iter += 1

        x = x + step_size * newton_direction
        indx_iter += 1

    if verbose:
        print("N_iter   = total number of iterations")
        print("N_func   = total number of function and gradient evaluations")
        print("F        = final function value \n")
        print("             * * *     \n")
        print("N_iter    N_func        F")
        print(f"{indx_iter + 1:6d}    {N_func:6d}    {loss.item():.6E}")
        print(f"  F = {loss.item():.12f} \n")

        if newton_decrement_square / 2.0 <= eps and indx_iter < max_iter:
            print("CONVERGENCE: 1/2*Newton_decrement^2 < EPS")
        else:
            print("CONVERGENCE: NUM_OF_ITERATION REACH MAX_ITERATION")

    return {
        "x": x,
        "N_iter": indx_iter + 1,
        "N_func": N_func,
        "F": loss.item(),
        "half_newton_decrement_square": newton_decrement_square / 2.0,
    }

def fmin_newton_fused(fgh, x_init, args=(), verbose=True, eps=1e-10, max_iter=500):
    """Newton's method with the same iterates as fmin_newton, but with a single
    function fgh(x, *args) -> (value, grad, hess) evaluated once per trial point.

    fmin_newton evaluates the value and gradient, the Hessian, and the value again
    in the line search, i.e. about three evaluations per iteration. Here the trial
    point of the line search is evaluated with fgh, and once it is accepted its
    gradient and Hessian are reused for the next iteration, so an iteration whose
    full step is accepted costs one evaluation. This matters when every evaluation
    is a pass over data streamed from the host.
    """

    ## constants used in backtracking line search; same as in fmin_newton
    alpha, beta = 0.01, 0.2
    max_ls_iter = 100

    if verbose:
        print("==================================================================")
        print("            RUNNING THE NEWTON'S METHOD (FUSED)                 \n")
        print("                           * * *                                \n")
        print(f"                    Tolerance EPS = {eps:.5E}                  \n")

    x = x_init
    loss, grad, H = fgh(x, *args)
    N_func = 1
    indx_iter = 0

    while indx_iter < max_iter:
        H = H + jnp.eye(H.shape[0]) * 1e-16
        newton_direction = jnp.linalg.solve(H, -grad)
        newton_decrement_square = jnp.sum(-grad * newton_direction)

        if verbose:
            print(
                f"At iterate {indx_iter:4d}; f= {loss.item():.5E};",
                f"|1/2*Newton_decrement^2|: {newton_decrement_square.item() / 2:.5E}\n",
            )

        if newton_decrement_square / 2.0 <= eps:
            break

        ## backtracking line search; each trial point is evaluated with fgh so
        ## that an accepted point already has its gradient and Hessian
        step_size = 1.0
        accepted = False
        for _ in range(max_ls_iter):
            trial = fgh(x + step_size * newton_direction, *args)
            N_func += 1
            approximate_loss = loss + step_size * alpha * (-newton_decrement_square)
            if trial[0] < approximate_loss:
                accepted = True
                break
            step_size = step_size * beta

        x = x + step_size * newton_direction
        if accepted:
            loss, grad, H = trial
        else:
            ## as in fmin_newton, take the (tiny) step anyway; it was not
            ## evaluated, so evaluate it now
            loss, grad, H = fgh(x, *args)
            N_func += 1
        indx_iter += 1

    if verbose:
        print("N_iter   = total number of iterations")
        print("N_func   = total number of function evaluations (passes over data)")
        print("F        = final function value \n")
        print("             * * *     \n")
        print("N_iter    N_func        F")
        print(f"{indx_iter + 1:6d}    {N_func:6d}    {loss.item():.6E}")
        print(f"  F = {loss.item():.12f} \n")

        if newton_decrement_square / 2.0 <= eps and indx_iter < max_iter:
            print("CONVERGENCE: 1/2*Newton_decrement^2 < EPS")
        else:
            print("CONVERGENCE: NUM_OF_ITERATION REACH MAX_ITERATION")

    return {
        "x": x,
        "N_iter": indx_iter + 1,
        "N_func": N_func,
        "F": loss.item(),
        "half_newton_decrement_square": newton_decrement_square / 2.0,
    }


######################################
#### L-BFGS optimizer using optax ####
@partial(jit, static_argnames=("f", "max_iter", "tol"))
def _run_opt(init_params, args, f, max_iter, tol):
    ## f and the settings are static while the data in args are traced, so repeated
    ## solves with the same loss and array shapes (e.g., bootstrap replicates)
    ## reuse one compiled loop instead of recompiling a new closure every call.
    def fun(x):
        return f(x, *args)

    opt = optax.chain(_print_info(), optax.lbfgs())
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = optax.tree_utils.tree_get(state, "count")
        grad = optax.tree_utils.tree_get(state, "grad")
        err = optax.tree_utils.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state

class _InfoState(NamedTuple):
    iter_num: jax.typing.ArrayLike

def _print_info():
    def init_fn(params):
        del params
        return _InfoState(iter_num=0)

    def update_fn(updates, state, params, *, value, grad, **extra_args):
        del params, extra_args

        jax.debug.print(
            "At iterate {i:4d}; f = {v:.3E}; grad norm: {e:.3E}\n",
            i=state.iter_num,
            v=value,
            e=optax.tree_utils.tree_l2_norm(grad),
        )
        return updates, _InfoState(iter_num=state.iter_num + 1)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)

def fmin_lbfgs(f, x_init, args=(), verbose=True, tol=1e-8, max_iter=500):
    def fun(x):
        return f(x, *args)

    if verbose:
        print("==================================================================")

        print("                RUNNING THE L-BFGS METHOD                     \n")
        print("                           * * *                              \n")
        print(f"                    Tolerance = {tol:.3E}                    \n")

    x_final, state_final = _run_opt(x_init, args, f=f, max_iter=max_iter, tol=tol)

    iter_num = optax.tree_utils.tree_get(state_final, "count")
    grad = optax.tree_utils.tree_get(state_final, "grad")
    grad_norm = optax.tree_utils.tree_l2_norm(grad)

    if verbose:
        print(
            f"At iterate {iter_num:4d}, f = {fun(x_final).item():.3E};",
            f"grad norm: {grad_norm.item():.3E}\n",
        )

        if grad_norm < tol and iter_num < max_iter:
            print("CONVERGENCE: GRADIENT NORM < TOL")
        else:
            print("CONVERGENCE: NUM_OF_ITERATION REACH MAX_ITERATION")
        
        print("==================================================================")

    return {
        "x": x_final,
        "N_iter": iter_num,
        "F": fun(x_final).item(),
        "grad_norm": grad_norm,
    }