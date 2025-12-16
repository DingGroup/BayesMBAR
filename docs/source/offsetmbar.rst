Offset-Constrained Coupled BayesMBAR: Mathematical Formulation and Implementation
==================================================================================

Abstract
--------

We present **OffsetMBAR**, a coupled Bayesian MBAR (Multistate Bennett Acceptance Ratio) formulation that enforces a linear constraint across multiple MBAR systems. Each system corresponds to a standard MBAR problem with its own set of thermodynamic states, reduced energies, and sample counts, but we require that the free energy of a particular state in each system, shifted by a given scalar offset, is identical across systems. We formalize this constraint, derive a linear parameterization in terms of a lower-dimensional parameter vector, and prove that the implementation correctly enforces the constraint while preserving the MBAR likelihood. The resulting optimization problem is solved by Newton's method or L-BFGS-B, and uncertainty is quantified by Hamiltonian Monte Carlo (HMC) sampling in the reduced parameter space.

----

1. Problem Setting
------------------

Consider :math:`S` MBAR systems, indexed by :math:`i = 0, \dots, S-1`. Each system has
:math:`M` thermodynamic states (assumed identical across systems):

- **Energies**: For system :math:`i`, we have reduced potentials :math:`u_{i,k}(\mathbf{x})` for states :math:`k = 0, \dots, M-1`, evaluated on a set of configurations (trajectories) with associated **counts**.
- **Counts**: Let :math:`N_{i,k}` be the number of configurations sampled from state :math:`k` in system :math:`i`. The code represents these as arrays ``energies[i]`` and ``nums_conf[i]``.

We denote by

.. math::

   F_{i,k}

the dimensionless free energy (in units of :math:`kT`) of state :math:`k` in system :math:`i`. As in standard MBAR, these free energies are identifiable only up to an additive constant per system, so it is natural to work with **free energy differences** relative to a reference state.

----

2. Classical MBAR Parameterization
----------------------------------

For each system :math:`i`, we define free energy differences relative to state 0:

.. math::

   \Delta F_{i,k} \equiv F_{i,k+1} - F_{i,0}, \quad k = 0,\dots,M-2.

Thus each system has :math:`M-1` independent parameters :math:`\Delta F_{i,0}, \dots, \Delta F_{i,M-2}`.

In vector form, for system :math:`i` we define

.. math::

   \mathbf{d}_i = 
   \begin{bmatrix}
   \Delta F_{i,0} \\
   \Delta F_{i,1} \\
   \vdots \\
   \Delta F_{i,M-2}
   \end{bmatrix}
   \in \mathbb{R}^{M-1}.

The associated **state free energies** can be reconstructed as

.. math::

   F_{i,0} = 0, \quad F_{i,k} = \Delta F_{i,k-1} \;\; \text{for } k=1,\dots, M-1.

i.e., the reference state 0 is fixed to zero, and all others are expressed as differences.

The standard MBAR log-likelihood (implemented via ``_compute_log_likelihood_of_dF``) is a function

.. math::

   \log \mathcal{L}_i(\mathbf{d}_i; \text{energies}_i, \text{nums_conf}_i),

and the joint log-likelihood across systems (without coupling) would be

.. math::

   \log \mathcal{L}(\{\mathbf{d}_i\}) = \sum_{i=0}^{S-1} \log \mathcal{L}_i(\mathbf{d}_i).

----

3. Offset Constraint Across Systems
-----------------------------------

We wish to **couple** these systems in a specific way. For each system :math:`i`, we are given a scalar offset :math:`\mathrm{offset}_i \in \mathbb{R}`. We select a particular thermodynamic state in each system—specifically, the **last MBAR state** (index :math:`M-1`)—and enforce that its free energy plus the corresponding offset is identical across systems.

Let us denote this last MBAR state free energy as

.. math::

   F_{i, M-1}.

The constraint imposed in the code (see ``_compute_offset_projection``) is:

.. math::

   F_{i,M-1} + \mathrm{offset}_i = c \quad \text{for all } i,

where :math:`c` is a shared constant (a new global parameter).

Since :math:`F_{i,M-1} = \Delta F_{i,M-2}` (because it is the last MBAR state above the reference 0), the constraint can be written equivalently as

.. math::

   \Delta F_{i,M-2} + \mathrm{offset}_i = c \quad \text{for all } i.

This defines a set of linear constraints tying together the last free-energy differences of each system.

----

4. Augmented State Space and Extra Offset State
-----------------------------------------------

The implementation further introduces an **extra offset state** per system. For system :math:`i`, the final "offset" state is defined to have free energy equal to the common constant :math:`c`. Intuitively:

- The **last MBAR state** (index :math:`M-1`) remains the physical state whose free energy satisfies :math:`F_{i,M-1} + \mathrm{offset}_i = c`.
- The **offset state** (index :math:`M`) is a synthetic state whose free energy is exactly :math:`c`, common to all systems.

Thus each system ends up with :math:`M+1` states when represented as ``F[i]`` in the code:

- :math:`F_{i,0} = 0` (reference),
- :math:`F_{i,1}, \dots, F_{i,M-1}` original MBAR states,
- :math:`F_{i,M}` additional offset state.

The constraint then becomes:

.. math::

   F_{i,M-1} + \mathrm{offset}_i = F_{i,M} = c, \quad \forall i.

This structure is reflected in ``_dF_to_state_F_with_offset``, where for each system segment:

- ``mbar_dF = dF[..., idx : idx + n - 1]`` holds the :math:`M-1` MBAR free energy differences :math:`\Delta F_{i,0}, \dots, \Delta F_{i,M-2}`,
- ``offset_value = dF[..., idx + n - 1]`` holds the extra offset state free energy, conceptually equal to :math:`c`.

The function reconstructs the full vector of free energies for system :math:`i` as

.. math::

   \mathbf{F}_i =
   \begin{bmatrix}
   0 \\
   \mathbf{mbar\_dF}_i \\
   \mathrm{offset\_value}_i
   \end{bmatrix}
   =
   \begin{bmatrix}
   0 \\
   \Delta F_{i,0} \\
   \vdots \\
   \Delta F_{i,M-2} \\
   c
   \end{bmatrix}
   \in \mathbb{R}^{M+1}.

----

5. Linear Parameterization via Projection Matrix
------------------------------------------------

The central mathematical device is the function ``_compute_offset_projection``, which constructs a **projection matrix** :math:`Q` and offset vector :math:`b` such that

.. math::

   \mathbf{d} = Q \mathbf{x} + \mathbf{b},

where:

- :math:`\mathbf{d}` is the **concatenated vector** of length :math:`S \cdot M`, containing for each system:

  - The :math:`M-1` MBAR free energy differences,
  - One additional scalar representing the offset state value.

- :math:`\mathbf{x} \in \mathbb{R}^{n_{\text{indep}}}` is a lower-dimensional vector of **independent parameters**:

  .. math::

     n_{\text{indep}} = S (M-2) + 1.

- :math:`Q \in \mathbb{R}^{S M \times n_{\text{indep}}}` is a full-rank linear map encoding both the unconstrained and constrained components.
- :math:`\mathbf{b} \in \mathbb{R}^{S M}` encodes the contribution from the known offsets :math:`\mathrm{offset}_i`.

5.1. Independent Parameters :math:`\mathbf{x}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each system :math:`i`, we treat the **first** :math:`M-2` **MBAR differences** as unconstrained:

.. math::

   x_{(i,j)} = \Delta F_{i,j}, \quad j = 0, \dots, M-3.

This contributes :math:`S (M-2)` free parameters. In addition, we introduce a global scalar:

.. math::

   x_{c} = c,

the shared constant appearing in the constraint.

Collecting all of these, we obtain :math:`\mathbf{x} \in \mathbb{R}^{S(M-2)+1}`.

5.2. Definition of :math:`Q` and :math:`b`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For system :math:`i` and its block within :math:`\mathbf{d}`, the code enforces:

1. **First** :math:`M-2` **MBAR differences are identity**:

   .. math::

      \Delta F_{i,j} = x_{(i,j)}, \quad j=0,\dots,M-3.

   In matrix form, this means that for corresponding indices,

   .. math::

      Q[\text{dF_idx} + j, \text{x_idx} + j] = 1,\quad b[\text{dF_idx} + j] = 0.

2. **Last MBAR difference** :math:`\Delta F_{i,M-2}` **satisfies the constraint**:

   We know

   .. math::

      \Delta F_{i,M-2} + \mathrm{offset}_i = c \Rightarrow
      \Delta F_{i,M-2} = c - \mathrm{offset}_i.

   Hence this component is *not* free; it is given by

   .. math::

      \Delta F_{i,M-2} = x_c - \mathrm{offset}_i.

   In matrix/vector terms:

   .. math::

      Q[\text{dF_idx} + M - 2, -1] = 1, \quad
      b[\text{dF_idx} + M - 2] = -\mathrm{offset}_i.

3. **Offset state has free energy** :math:`c`:

   The augmented offset state free energy is set to equal the same constant :math:`c`:

   .. math::

      F_{i,M} = c.

   As implemented, the offset state entry in :math:`\mathbf{d}` is simply equal to :math:`c`:

   .. math::

      \mathrm{offset\_value}_i = x_c.

   Hence for the entry at ``dF_idx + M - 1``:

   .. math::

      Q[\text{dF_idx} + M - 1, -1] = 1, \quad b[\text{dF_idx} + M - 1] = 0.

All other entries of :math:`Q` and :math:`b` are zero.

5.3. Proof that the Constraint is Enforced
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\mathbf{x}` be any vector of independent parameters, and let :math:`\mathbf{d} = Q\mathbf{x} + \mathbf{b}`. Partition :math:`\mathbf{d}` by system, and within each system, extract:

- MBAR differences :math:`\Delta F_{i,0}, \dots, \Delta F_{i,M-2}`,
- Offset state value :math:`F_{i,M}`.

From the construction:

1. For :math:`j = 0, \dots, M-3`:

   .. math::

      \Delta F_{i,j} = x_{(i,j)}.

2. For the **last MBAR difference**:

   .. math::

      \Delta F_{i,M-2} = x_c - \mathrm{offset}_i.

3. For the **offset state**:

   .. math::

      F_{i,M} = x_c.

Reconstruct physical free energies as in ``_dF_to_state_F_with_offset``:

- :math:`F_{i,0} = 0`,
- :math:`F_{i,k} = \Delta F_{i,k-1}` for :math:`k = 1,\dots, M-1`,
- :math:`F_{i,M} = x_c`.

Then the **last MBAR state** free energy is

.. math::

   F_{i,M-1} = \Delta F_{i,M-2} = x_c - \mathrm{offset}_i.

Therefore,

.. math::

   F_{i,M-1} + \mathrm{offset}_i = (x_c - \mathrm{offset}_i) + \mathrm{offset}_i = x_c,

which is independent of :math:`i`. This proves that for **any** :math:`\mathbf{x}`, the resulting free energies satisfy

.. math::

   F_{i,M-1} + \mathrm{offset}_i = c := x_c \quad \forall i,

and that the offset state :math:`F_{i,M} = c` is also identical across systems.

Thus, the linear map :math:`\mathbf{x} \mapsto (Q\mathbf{x} + \mathbf{b}) \mapsto \{\mathbf{F}_i\}_i` **exactly enforces** the specified offset constraint.

----

6. Likelihood and Loss Function
-------------------------------

Given :math:`\mathbf{x}`, the code computes:

1. The full free energy-difference vector:

   .. math::

      \mathbf{d}_{\text{full}} = Q\mathbf{x} + \mathbf{b}.

2. For each system :math:`i`, extracts the MBAR portion:

   .. math::

      \mathbf{d}_i = \mathbf{d}_{\text{full}}[i\text{-block}, \; :M-1].

   (i.e., the first :math:`M-1` entries in that system's block).

3. Computes the **MBAR log-likelihood** for each system, as a function of :math:`\mathbf{d}_i`, energies, and counts, via ``_compute_log_likelihood_of_dF``:

   .. math::

      \log \mathcal{L}_i(\mathbf{d}_i).

4. Sums across systems:

   .. math::

      \log \mathcal{L}(\mathbf{x}) =
      \sum_{i=0}^{S-1} \log \mathcal{L}_i\big(\mathbf{d}_i(\mathbf{x})\big).

This is implemented in ``_compute_offset_log_likelihood``.

Then ``_compute_offset_loss_likelihood`` defines the **loss function** as the (normalized) negative log-likelihood per configuration:

.. math::

   \mathcal{J}(\mathbf{x}) = - \frac{1}{N_{\text{tot}}} \log \mathcal{L}(\mathbf{x}),

where

.. math::

   N_{\text{tot}} = \sum_{i=0}^{S-1} \sum_{k=0}^{M-1} N_{i,k}

is the total number of samples across all systems. This normalization does not change the optimizer's argmin, only the scale of the objective.

**Key point:** Because the map :math:`\mathbf{x} \mapsto \{\mathbf{d}_i\}` is linear and injective (ignoring the usual additive reference freedom, which is fixed by setting :math:`F_{i,0}=0`), optimizing :math:`\mathcal{J}(\mathbf{x})` is equivalent to optimizing the MBAR log-likelihood in the constrained subspace.

----

7. Optimization and Mode Finding
--------------------------------

The constructor of ``OffsetMBAR`` proceeds as follows:

1. Initializes :math:`\mathbf{x}_0 = 0` (a zero vector of appropriate dimension).
2. Chooses a method:

   - **Newton**: uses a JAX-jitted function computing value and gradient of :math:`\mathcal{J}`, and its Hessian, then calls ``fmin_newton`` to find a stationary point.
   - **L-BFGS-B**: uses SciPy's ``optimize.minimize`` with objective and gradient computed via JAX (``value_and_grad``).

Let :math:`\mathbf{x}^*` denote the optimizer found by either method. Then:

- The **mode of the constrained likelihood** in the :math:`\mathbf{x}`-space is :math:`\mathbf{x}^*`.
- The corresponding **mode in the** :math:`\mathbf{d}`-**space** is

  .. math::

     \mathbf{d}^* = Q\mathbf{x}^* + \mathbf{b}.

- The state free energies per system are computed via ``_dF_to_state_F_with_offset``:

  .. math::

     \{\mathbf{F}_i^*\} = \text{_dF_to_state_F_with_offset}(\mathbf{d}^*, \text{nums_state}).

These are stored as:

- ``self._dF_mode_ll`` = :math:`\mathbf{d}^*`,
- ``self._state_F_mode_ll`` = :math:`\{\mathbf{F}_i^*\}`.

The public property ``F_mode`` converts these JAX arrays to NumPy on CPU.

By construction (Section 5.3), :math:`\{\mathbf{F}_i^*\}` exactly satisfy

.. math::

   F^*_{i,M-1} + \mathrm{offset}_i = c^*

for all :math:`i`, where :math:`c^* = x_c^*`.

----

8. Posterior Sampling via HMC
-----------------------------

If ``sample_size > 0``, the implementation samples from the **posterior (or likelihood) in** :math:`\mathbf{x}`-**space** around the mode:

1. Defines

   .. math::

      \log p(\mathbf{x}) := \log \mathcal{L}(\mathbf{x}) 
      = \text{_compute_offset_log_likelihood}(\mathbf{x}, Q, b, \text{energies}, \text{nums_conf}).

   (Strictly speaking, this is the log-likelihood; one can interpret it as log-posterior if combined with a flat prior.)

2. Uses ``_sample_from_logdensity`` (an HMC-based sampler from ``bayesmbar.py``) with:

   - Initial position :math:`\mathbf{x}_0 = \mathbf{x}^*`,
   - Log-density function :math:`\log p`,
   - Warmup steps and number of samples specified.

This yields a set of samples :math:`\{\mathbf{x}^{(s)}\}_{s=1}^S` from an approximation to the posterior over :math:`\mathbf{x}`. Each is mapped to:

- Free energy differences:

  .. math::

     \mathbf{d}^{(s)} = Q\mathbf{x}^{(s)} + \mathbf{b}.

- System-wise free energies:

  .. math::

     \{\mathbf{F}_i^{(s)}\}.

The properties:

- ``F_samples`` gives :math:`\{\mathbf{F}_i^{(s)}\}`,
- ``F_mean`` gives their sample mean,
- ``DeltaF_mode``, ``DeltaF_mean``, and ``DeltaF_std`` report derived statistics of pairwise free-energy differences within each system.

Each sampled :math:`\mathbf{F}_i^{(s)}` inherits the linear constraint (Section 5.3), so the offset condition holds **for all samples**, not only for the mode.

----

9. Summary of Mathematical Soundness of the Implementation
----------------------------------------------------------

- The **constraint** :math:`F_{i,M-1} + \mathrm{offset}_i = c` is encoded as a **linear condition** in the free energy differences :math:`\mathbf{d}`.
- A full-rank **projection** :math:`(Q,b)` is constructed such that any :math:`\mathbf{x}` yields a :math:`\mathbf{d} = Q\mathbf{x} + \mathbf{b}` satisfying the constraint exactly.
- The **MBAR likelihood** is evaluated only on the physical MBAR states (excluding the synthetic offset state), ensuring that the constraint does not artificially alter the likelihood contributions.
- **Optimization** in :math:`\mathbf{x}`-space is equivalent to maximizing the MBAR likelihood in the constrained subspace of :math:`\mathbf{d}`-space.
- **Sampling** in :math:`\mathbf{x}`-space, using the same projection, produces uncertainty estimates that are consistent with both the MBAR likelihood and the offset constraint.

Taken together, these components form a coherent and mathematically consistent realization of the offset-constrained coupled BayesMBAR model.

