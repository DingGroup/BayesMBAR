import jax.numpy as jnp
from pytest import approx

from bayesmbar.utils import (
    _hessian_loss_likelihood_of_dF,
    _value_and_grad_loss_likelihood_of_dF,
    fmin_newton,
    fmin_newton_fused,
)


def test_fmin_newton_fused_matches_fmin_newton(setup_mbar_data):
    energy, num_conf, F_ref, energy_p, F_ref_p = setup_mbar_data
    energy, num_conf = jnp.asarray(energy), jnp.asarray(num_conf, dtype=float)
    x0 = jnp.zeros(energy.shape[0] - 1)

    def fgh(x, energy, num_conf):
        value, grad = _value_and_grad_loss_likelihood_of_dF(x, energy, num_conf)
        return value, grad, _hessian_loss_likelihood_of_dF(x, energy, num_conf)

    ref = fmin_newton(
        _value_and_grad_loss_likelihood_of_dF,
        _hessian_loss_likelihood_of_dF,
        x0,
        args=(energy, num_conf),
        verbose=False,
    )
    res = fmin_newton_fused(fgh, x0, args=(energy, num_conf), verbose=False)

    ## same iterates, fewer evaluations
    assert res["N_iter"] == ref["N_iter"]
    assert res["x"] == approx(ref["x"], abs=1e-12)
    assert res["N_func"] < ref["N_func"] + ref["N_iter"]
