Adding an equilibrium profile or elliptic scheme
=================================================

The ``physics/equilibrium`` slice solves the Grad-Shafranov equation

.. math::

   \Delta^* \psi \equiv r \partial_r\!\left(\frac{1}{r}\partial_r \psi\right)
   + \partial_{zz} \psi
   = -\mu_0 r^2 p'(\psi) - F F'(\psi)

for a free-boundary axisymmetric MHD equilibrium. It is the framework's only
**boundary-value** problem -- every other Quasar solver is an explicit
hyperbolic time-marching scheme -- so it introduces an elliptic numerics stack
that has no analogue elsewhere in the tree.

This guide covers the two extension points: equilibrium **profiles** (the free
functions :math:`p'` and :math:`FF'`) and the **elliptic operator/solver** axis.

Architecture at a glance
------------------------

Generic, reusable elliptic machinery lives under ``numerics/``; the
GS-specific physics lives under ``physics/equilibrium/``:

===================================================  ==========================================================
Header                                               Responsibility
===================================================  ==========================================================
``numerics/elliptic_grid.hpp``                       Node-centered ``(r,z)`` grid and multigrid hierarchy
``numerics/pade_derivative.hpp``                     Sixth-order compact derivative coefficients and closures
``numerics/pade_line_solve.hpp``                     Pivoting tridiagonal line solve (host reference)
``numerics/gs_operator_l2.hpp``                      Second-order smoothable operator (multigrid workhorse)
``numerics/gs_operator_l6.hpp``                      Sixth-order operator (residual evaluation only)
``numerics/geometric_multigrid.hpp``                 Matrix-free V-cycle / FMG
``numerics/defect_correction.hpp``                   Couples L6 accuracy to L2 solvability
``numerics/elliptic_tile.hpp``                       Tile decomposition and halo contract
``physics/equilibrium/equilibrium_profile.hpp``      ``IEquilibriumProfile``, polynomial family, Solov'ev
``physics/equilibrium/critical_points.hpp``          Magnetic axis / X-point location
``physics/equilibrium/free_boundary.hpp``            Green's function, coil field, boundary integral
``physics/equilibrium/gs_solver.hpp``                Nonlinear outer loop and failure contract
``physics/equilibrium/kernels.hpp``                  Device kernel-launch ABI for the whole module
``physics/equilibrium/flux_surfaces.hpp``            ``B``, ``q(\psi)``, surface geometry and metrics
``physics/equilibrium/mhd_seeding.hpp``              Projection onto the staggered MHD mesh
===================================================  ==========================================================

Adding a profile
----------------

Implement ``IEquilibriumProfile`` (four methods: ``dp_dpsi``, ``ff_prime``, and
their second derivatives) and pass it to ``GsSolver``. Profiles are functions of
the **normalized** flux :math:`\psi_N = (\psi-\psi_{axis})/(\psi_b-\psi_{axis})`,
which is zero on the magnetic axis and one on the plasma boundary -- the
EFIT/FreeGS convention, so a polynomial profile is directly comparable with
published equilibria.

.. important::

   The solver runs on the GPU, and a virtual interface cannot cross to the
   device: a vtable pointer is only valid for an object constructed on the side
   that dispatches through it. ``IEquilibriumProfile`` therefore stays entirely
   host-side, and ``GsSolver`` lowers the selected profile to a flat
   ``ProfileCoefficients`` POD at construction, which the kernels evaluate with
   no indirection.

   The consequence is concrete: **only ``PolynomialProfile`` currently works.**
   Passing anything else is a construction-time ``std::invalid_argument``,
   deliberately, rather than a silent fallback to something slower or wrong.

   Adding a non-polynomial profile therefore takes one extra step beyond
   implementing the interface -- giving it a device representation. For a spline
   or tabulated form the natural route is a sampled table plus a tag on
   ``ProfileCoefficients``, with the ``__device__`` evaluator dispatching on the
   tag. Implement the host interface as well: it is what the registry, the deck,
   and the per-kernel reference comparisons use.

Two further requirements are easy to miss:

* **Vanish at the boundary.** ``dp_dpsi(1)`` and ``ff_prime(1)`` should be zero,
  otherwise current is driven outside the last closed flux surface.
* **Supply real second derivatives.** They feed the Newton Jacobian. A finite
  difference works but degrades the Newton phase.

The amplitude of the profile is **not** free: the solver rescales both free
functions every outer iteration so the integrated toroidal current matches the
requested :math:`I_p`. Without that normalization the problem is ill-posed.
The applied factor is returned as ``GsResult::profile_scale`` and must be passed
to ``integrate_f_profile``; integrating the raw :math:`FF'` would reconstruct a
toroidal field and safety factor inconsistent with the solved equilibrium.

Six lessons the implementation paid for
---------------------------------------

These are non-obvious and each cost real debugging time. They are documented at
length in the corresponding headers.

**1. Compact-scheme order conditions are degenerate.** For the first derivative
every even-order condition vanishes identically; for the second derivative every
odd one does. Solving "the first three conditions" in index order silently
yields the *fourth*-order scheme (:math:`\alpha=1/4`, :math:`1/10`) rather than
the sixth-order one (:math:`\alpha=1/3`, :math:`2/11`). A test pins the
constants for exactly this reason.

**2. Near-boundary rows cannot be chosen independently.** Patching node 1 with
the textbook fourth-order compact pair degraded node 0 to ~3.5 order -- *below*
the order of the row that caused it -- because the node-0 closure assumes a
sixth-order implicit neighbour. Every boundary-adjacent row must close at the
target order together.

**3. These closures require a pivoting line solve.** The boundary rows are not
diagonally dominant (:math:`\alpha = 5`, :math:`126/11`, :math:`-131/22`).
Unpivoted Thomas hits an *exact zero pivot* on the second-derivative system and
the elimination factor explodes to ~8e14. The matrix is fine (condition number
3.3e4, grid-independent); the algorithm was not. **A textbook pivot-free
parallel cyclic reduction is therefore invalid for the device port.**

**4. A sixth-order second derivative saturates fp64 almost immediately.**
Forming :math:`\Delta^*` divides by :math:`h^2`, so the error floor grows as
:math:`\epsilon/h^2` while truncation falls as :math:`h^6`. On a unit domain the
crossover is near ``n = 33``. An order study on a smooth solution past that point
measures the *floor* -- apparent rate ~3.5, then negative. Use a
higher-wavenumber manufactured solution to measure order.

**5. A vacuum field never contains a magnetic axis.** :math:`\Delta^*\psi = 0`
admits no interior extremum, so the coil field alone has no O-point -- and
without an axis there is no :math:`\psi_N`, hence no current, hence no axis. The
solver must seed a plasma column to bootstrap. Seed the well **with the sign the
current will drive** (:math:`\psi` is a *maximum* on axis for positive
:math:`I_p`) and at a depth scaled to :math:`\mu_0 I_p`, not to the vacuum
field's own variation.

**6. Newton is implemented but disabled by default.** The diagonal profile
Jacobian is the standard fixed-boundary term; on a *free*-boundary problem the
boundary condition is itself a dense functional of the interior :math:`\psi`
through the Green's-function integral. With that block missing the Newton
direction is wrong by an O(1) amount near convergence. Measured on the reference
case: Picard converges in 222 iterations, Newton stalls at 7.4e-4. Completing it
requires a Jacobian-free Newton-Krylov formulation or the von Hagenow surface-
current form. See ``GsConfig::enable_newton``.

Failure is an answer
--------------------

A free-boundary equilibrium may legitimately not exist. ``GsResult`` therefore
carries a ``GsStatus`` (``no_closed_surface``, ``axis_lost``,
``critical_point_overflow``, ``numerical_failure``, ``residual_stalled``,
``iteration_limit``) plus the
best-effort :math:`\psi`, rather than throwing -- the coil-design optimizer must
be able to score a failed configuration and continue. Flux-surface diagnostics
follow the same rule:
open surfaces remain available with ``closed == false`` and are counted by
``n_open_surfaces``, but are excluded from the q profile and aggregate geometry;
``psi_n_boundary`` identifies the outermost closed surface used for those
aggregates. Malformed *input* still throws.

Feeding the MHD solver
----------------------

``mhd_seeding.hpp`` projects :math:`\psi` onto the staggered MHD mesh. It
interpolates :math:`\psi` to cell corners and **differences** it to obtain the
face fields, rather than interpolating :math:`B` directly. Because both in-plane
components derive from one scalar potential by a discrete curl, the discrete
divergence telescopes to zero to round-off (measured: 1.3e-13 relative). This
matters because ``MhdSolver2D::seed_background`` rejects a non-solenoidal
background, and constrained transport preserves whatever divergence the initial
state carries. The GS grid is strictly annular, so projection also requires the
MHD target grid and its ghost faces to remain at positive radius. A target that
reaches the cylindrical axis is rejected rather than assigned an invented
regularity continuation that would break discrete solenoidality.

Distributed execution
---------------------

``elliptic_tile.hpp`` provides the tile decomposition. The second-order path is
tile-local and its tiled result is bit-for-bit identical to the serial one.

The sixth-order path is **not** tile-local: a compact derivative is an implicit
solve along an entire grid line, so no finite halo suffices. Three strategies are
viable -- slab decomposition with a transpose, a distributed tridiagonal solve
(which must pivot, per lesson 3), or tile-level defect correction that gives up
uniform sixth order. ``TileGrid::supports_local_pade()`` reports whether a given
tile may run the operator locally, so callers assert the precondition instead of
silently computing a wrong answer.

Verification
------------

.. code-block:: bash

   ctest --preset hip-gfx950-release -R "numerics_test_(pade|geometric|defect|elliptic)"
   ctest --preset hip-gfx950-release -R "^equilibrium_"

The manufactured-solution tests establish scheme order; the Solov'ev tests
establish that the physics and sign conventions are right. Note that Solov'ev is
a low-degree polynomial that a sixth-order scheme reproduces exactly, so it
**cannot** measure convergence order -- that is what the MMS tests are for.
