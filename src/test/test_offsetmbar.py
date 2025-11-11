import numpy as np
import pytest
from bayesmbar import CBayesMBAR, OffsetMBAR
import numpy.testing as npt

n_samples = 100


@pytest.fixture()
def no_offset():
    M = 5  ## number of states
    repeats = 3
    mu = np.linspace(0, 1, M)  ## equilibrium positions
    k = np.random.uniform(10, 30, M)  ## force constants
    sigma = np.sqrt(1.0 / k)
    F_reference = -np.log(sigma)
    F_reference -= F_reference[0]
    n = n_samples
    energies = []
    for _ in range(repeats):
        x = [np.random.normal(mu[i], sigma[i], (n,)) for i in range(M)]
        x = np.concatenate(x)
        u = 0.5 * k.reshape((-1, 1)) * (x - mu.reshape((-1, 1))) ** 2
        energies.append(u)
    num_conf = [np.array([n for i in range(M)])] * repeats
    return F_reference, energies, num_conf, repeats


def test_cmbar(no_offset):
    F_reference, energies, num_conf, repeats = no_offset
    cmbar = CBayesMBAR(
        energies,
        num_conf,
        identical_states=[
            [(i, 0) for i in range(repeats)],
            [(i, 4) for i in range(repeats)],
        ],
        sample_size=1000,
        warmup_steps=100,
        random_seed=0,
        verbose=False,
    )
    f_mean = cmbar.F_mean

    assert f_mean[0][-1] == pytest.approx(F_reference[-1], abs=0.2)
    f = [d_f[-1] for d_f in f_mean]
    npt.assert_allclose(f, f[0], atol=1e-5)


def test_0offset(no_offset):
    F_reference, energies, num_conf, repeats = no_offset
    cmbar = OffsetMBAR(
        energies,
        num_conf,
        offsets=[0] * repeats,
        sample_size=1000,
        warmup_steps=100,
        random_seed=0,
        verbose=False,
    )

    f_mode = cmbar.F_mode
    f = [d_f[-1] for d_f in f_mode]
    npt.assert_allclose(f, f[0], atol=0.1)
    for edge in f_mode:
        assert edge[-1] - edge[-2] == pytest.approx(0, abs=0.1)
    assert f_mode[0][-1] == pytest.approx(F_reference[-1], abs=0.2)


def test_offset():
    M = 5  ## number of states
    repeats = 3
    mu = np.linspace(0, 1, M)  ## equilibrium positions
    n = n_samples
    F_reference_list = []
    energies = []
    for _ in range(repeats):
        k = np.random.uniform(10, 30, M)  ## force constants
        sigma = np.sqrt(1.0 / k)
        F_reference = -np.log(sigma)
        F_reference -= F_reference[0]
        F_reference_list.append(F_reference)

        x = [np.random.normal(mu[i], sigma[i], (n,)) for i in range(M)]
        x = np.concatenate(x)
        u = 0.5 * k.reshape((-1, 1)) * (x - mu.reshape((-1, 1))) ** 2
        energies.append(u)
    offsets = [-1 * F_reference[-1] for F_reference in F_reference_list]
    num_conf = [np.array([n for i in range(M)])] * repeats
    cmbar = OffsetMBAR(
        energies,
        num_conf,
        offsets=offsets,
        sample_size=1000,
        warmup_steps=100,
        random_seed=0,
        verbose=False,
    )

    f_mode = cmbar.F_mode
    for edge, offset in zip(f_mode, offsets):
        assert edge[0] == pytest.approx(0, abs=0.1)
        assert edge[-1] == pytest.approx(0, abs=0.1)
        assert edge[-1] - edge[-2] == offset
