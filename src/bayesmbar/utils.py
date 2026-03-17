import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax import hessian
from scipy import optimize
from jax.scipy.special import logsumexp
import optax
from typing import NamedTuple


def _solve_mbar(dF_init, energy, num_conf, method, verbose):
    if method == "Newton":
        f = jit(value_and_grad(_compute_loss_likelihood_of_dF))
        hess = jit(hessian(_compute_loss_likelihood_of_dF))
        res = fmin_newton(f, hess, dF_init, args=(energy, num_conf), verbose=verbose)
        dF = res["x"]
    elif method == "L-BFGS":
        res = fmin_lbfgs(
            _compute_loss_likelihood_of_dF,
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
def _run_opt(init_params, fun, opt, max_iter, tol):
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
    opt = optax.chain(_print_info(), optax.lbfgs())

    def fun(x):
        return f(x, *args)

    if verbose:
        print("==================================================================")

        print("                RUNNING THE L-BFGS METHOD                     \n")
        print("                           * * *                              \n")
        print(f"                    Tolerance = {tol:.3E}                    \n")

    x_final, state_final = _run_opt(x_init, fun, opt, max_iter=max_iter, tol=tol)

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