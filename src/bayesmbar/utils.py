import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax import hessian
from scipy import optimize
from jax.scipy.special import logsumexp

def _solve_mbar(dF_init, energy, num_conf, method, verbose=False):
    if method == "Newton":
        f = jit(value_and_grad(_compute_loss_likelihood_of_dF))
        hess = jit(hessian(_compute_loss_likelihood_of_dF))
        res = fmin_newton(f, hess, dF_init, args=(energy, num_conf))
        dF = res["x"]
    elif method == "L-BFGS-B":
        options = {"disp": verbose, "gtol": 1e-8}
        f = jit(value_and_grad(_compute_loss_likelihood_of_dF))
        results = optimize.minimize(
            lambda x: [np.array(r) for r in f(x, energy, num_conf)],
            dF_init,
            jac=True,
            method="L-BFGS-B",
            tol=1e-12,
            options=options,
        )
        dF = results["x"]
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


def fmin_newton(f, hess, x_init, args=(), verbose=True, eps=1e-10, max_iter=300):
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

        newton_direction = np.linalg.solve(H, -grad)
        newton_decrement_square = np.sum(-grad * newton_direction)

        if verbose:
            print(
                f"At iterate {indx_iter:4d}; f= {loss.item():.5E};",
                f"|1/2*Newton_decrement^2|: {newton_decrement_square.item()/2:.5E}\n",
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
        print(f"{indx_iter+1:6d}    {N_func:6d}    {loss.item():.6E}")
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
