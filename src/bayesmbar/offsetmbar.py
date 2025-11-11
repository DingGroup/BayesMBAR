import numpy as np

from bayesmbar import CBayesMBAR
from numpy import ndarray
from loguru import logger

class OffsetMBAR:
    def __init__(
        self,
        energies: list[np.ndarray],
        nums_conf: list[np.ndarray],
        offsets: list[float],
        sample_size: int = 1000,
        warmup_steps: int = 500,
        method: str = "Newton",
        random_seed: int = None,
        verbose: bool = True,
    ) -> None:
        """
        Offset-constrained Coupled BayesMBAR.

        This variant of Coupled BayesMBAR enforces a simple constraint: for each
        coupled system i, the free-energy difference computed from its energies
        plus a provided scalar offset must be identical across all systems. In
        other words, the solution to the MBAR equation satisfies
        mbar(energies[i]) + offsets[i] = constant for every system i.

        Parameters
        ----------
        energies : List[numpy.ndarray]
            Per-system arrays of reduced potentials (in units of kT) used by
            MBAR. One array per coupled system.
        nums_conf : List[numpy.ndarray]
            Per-system arrays of counts (number of configurations) corresponding
            to the provided energies.
        offsets : List[float]
            One scalar offset per system. The constraint is that
            mbar(energies[i]) + offsets[i] is the same across all systems.
        sample_size : int, optional
            Number of samples to draw from the likelihood. Default: 1000.
        warmup_steps : int, optional
            Number of warmup steps for the HMC sampler. Default: 500.
        method : str, optional
            Optimization method to find the likelihood mode. Either "Newton" or
            "L-BFGS-B". Default: "Newton".
        random_seed : int, optional
            Random seed. If None, a seed is generated from the current time.
            Default: None.
        verbose : bool, optional
            If True, print sampling progress. Default: True.
        """
        new_energies = []
        new_nums_conf = []
        first_state = []
        last_state = []
        connecting_states = []
        index = 0
        for energy, n_conf, offset in zip(energies, nums_conf, offsets, strict=True):
            # Add the original reduced potential
            states, n_samples = energy.shape
            new_energies.append(energy)
            new_nums_conf.append(n_conf)
            first_state.append((index, 0))
            # Generate the reduced potential for offset
            index += 1
            # Make it divisible
            slice = energy[0, :n_samples]
            new_energy = np.linspace(0, offset, states).reshape(
                (states, 1)
            ) + slice.reshape((1, n_samples))
            new_energies.append(new_energy)
            new_nums_conf.append(n_conf)
            last_state.append((index, states - 1))
            connecting_states.append([(index - 1, states - 1), (index, 0)])
            index += 1

        logger.info(f"identical_states: {[first_state, last_state, *connecting_states]}")
        for i, (energy, nums_conf) in enumerate(zip(new_energies, new_nums_conf)):
            logger.info(f"Edge {i}: {nums_conf=} energy:({energy.shape})")

        self.cbmbar = CBayesMBAR(
            new_energies,
            new_nums_conf,
            identical_states=[first_state, last_state, *connecting_states],
            method=method,
            sample_size=sample_size,
            warmup_steps=warmup_steps,
            random_seed=random_seed,
            verbose=verbose,
        )

    @property
    def F_mode(self) -> list[ndarray]:
        F_mode_list = []
        for real_f, offset_f in zip(self.cbmbar.F_mode[::2], self.cbmbar.F_mode[1::2]):
            F_mode = np.append(real_f, real_f[-1] + offset_f[-1] - offset_f[0])
            F_mode_list.append(F_mode)
        return F_mode_list

    @property
    def F_mean(self) -> list[ndarray]:
        print(self.cbmbar.F_mean)
        F_mean_list = []
        for real_f, offset_f in zip(self.cbmbar.F_mean[::2], self.cbmbar.F_mean[1::2]):
            F_mean = np.append(real_f, real_f[-1] + offset_f[-1] - offset_f[0])
            F_mean_list.append(F_mean)
        return F_mean_list
