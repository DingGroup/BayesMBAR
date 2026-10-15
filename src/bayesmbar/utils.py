from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import hessian, jit, value_and_grad
from jax.scipy.special import logsumexp


def _solve_mbar(dF_init, energy, num_conf, method, verbose, batch_size=None):
    """Solve the MBAR equations for dF.

    If batch_size is None, the loss, gradient and Hessian are computed over the whole
    energy matrix at once. Otherwise they are accumulated over batches of
    batch_size configurations, so that the temporary memory scales with batch_size
    instead of with the number of configurations.
    """
    if batch_size is None:
        loss = _compute_loss_likelihood_of_dF
        value_and_grad_fn = _value_and_grad_loss_likelihood_of_dF
        hessian_fn = _hessian_loss_likelihood_of_dF
    else:
        ## put the energy matrix on the device once instead of in every call;
        ## jax.device_put avoids the transient second copy made by jnp.asarray
        energy = jax.device_put(energy)
        num_conf = jnp.asarray(num_conf, dtype=energy.dtype)
        loss, value_and_grad_fn, hessian_fn = _batched_loss_functions(batch_size)

    if method == "Newton":
        res = fmin_newton(
            value_and_grad_fn,
            hessian_fn,
            dF_init,
            args=(energy, num_conf),
            verbose=verbose,
        )
        dF = res["x"]
    elif method == "L-BFGS":
        res = fmin_lbfgs(
            loss,
            dF_init,
            args=(energy, num_conf),
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


def _accumulate_in_batches(stats, energy, batch_size, init):
    """Sum stats(energy[:, batch]) over batches of batch_size columns of energy.

    stats maps an (M, B) block of the energy matrix to a pytree with the same
    structure as init. Full batches are processed in a fori_loop so that only one
    batch of intermediates is alive at a time; the remaining N % batch_size columns
    are processed once at the end.
    """
    N = energy.shape[1]
    ## the loop body is traced even when it runs zero times, so the slice in it
    ## must fit in the energy matrix
    batch_size = min(batch_size, N)
    num_full_batches = N // batch_size

    def body(i, acc):
        u = jax.lax.dynamic_slice_in_dim(energy, i * batch_size, batch_size, axis=1)
        return jax.tree.map(jnp.add, acc, stats(u))

    acc = jax.lax.fori_loop(0, num_full_batches, body, init)
    if N % batch_size:
        acc = jax.tree.map(
            jnp.add, acc, stats(energy[:, num_full_batches * batch_size :])
        )
    return acc


def _loss_grad_hess_in_batches(dF, energy, num_conf, batch_size, with_hessian):
    """Loss, gradient and (optionally) Hessian of _compute_loss_likelihood_of_dF,
    accumulated over batches of configurations.

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

    def stats(u):
        logits = F + logn - u.T
        lse = logsumexp(logits, axis=1)
        p = jnp.exp(logits - lse[:, None])
        s = (lse.sum(), p.sum(axis=0))
        if with_hessian:
            s = s + (p.T @ p,)
        return s

    init = (jnp.zeros((), energy.dtype), jnp.zeros((M,), energy.dtype))
    if with_hessian:
        init = init + (jnp.zeros((M, M), energy.dtype),)
    sums = _accumulate_in_batches(stats, energy, batch_size, init)

    N = num_conf.sum()
    loss = (sums[0] - jnp.sum(num_conf * F)) / N
    grad = (sums[1] - num_conf)[1:] / N
    if not with_hessian:
        return loss, grad
    hess = (jnp.diag(sums[1]) - sums[2])[1:, 1:] / N
    return loss, grad, hess


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
    def loss(dF, energy, num_conf):
        return _loss_grad_hess_in_batches(dF, energy, num_conf, batch_size, False)[0]

    def loss_fwd(dF, energy, num_conf):
        value, grad = _loss_grad_hess_in_batches(
            dF, energy, num_conf, batch_size, False
        )
        return value, grad

    def loss_bwd(grad, g):
        ## only dF is differentiated; None means a zero cotangent for the data
        return (g * grad, None, None)

    loss.defvjp(loss_fwd, loss_bwd)

    @jit
    def value_and_grad_fn(dF, energy, num_conf):
        return _loss_grad_hess_in_batches(dF, energy, num_conf, batch_size, False)

    @jit
    def hessian_fn(dF, energy, num_conf):
        return _loss_grad_hess_in_batches(dF, energy, num_conf, batch_size, True)[2]

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