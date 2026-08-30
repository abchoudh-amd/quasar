Extending the ideal-MHD stability slice
=======================================

This page is the source of truth for changing ``physics/stability``. Unlike the
other dev-guide pages it does not describe a plugin registry: the stability
slice has no string-selected components yet, because it has exactly one energy
functional and one discretization. What it has instead is a chain of contracts
that each stage relies on the previous one to have established, and the point of
this page is to say what those are before you change one of them.

Read :doc:`../theory/toroidal_ideal_mhd_energy` first. The continuum conventions
there -- the sign of ``theta``, the ``exp[i(m theta - n phi)]`` dependence, and
the realification of a Hermitian entry -- are part of the ABI, not of the
derivation, and code that disagrees with them will produce a plausible spectrum
that is wrong.

The pipeline
------------

``StabilitySolver::solve_mode`` runs these stages in order. Every one of them can
fail in a way that is *reported* rather than thrown, through ``ModeSolveStatus``:

1. **Prepare the equilibrium.** Integrate the ``F`` profile, form the field and
   its derivatives, and trace probe surfaces to sample ``q``. Failure:
   ``surface_trace_failed``.
2. **Locate rational surfaces** for this ``n`` and lay out radial domains cut at
   them. Failure: ``topology_overflow``, or ``unsupported_rational_topology``
   whenever any resonance survives the filter at all.
3. **Build the Chebyshev basis** on each subdomain and trace one contour per
   Lobatto node, then build PEST coordinates on that tensor-product grid.
4. **Build the toroidal equilibrium fields** on the same grid. Failure:
   ``equilibrium_fields_failed``.
5. **Validate the geometry**: the field-line pitch must equal ``q``, ``J
   B^theta`` must equal the flux scale, and ``J / R^2`` must be a flux function.
   Failure: ``geometry_validation_failed``.
6. **Assemble** the energy and inertia into a dense Hermitian pencil. Failure:
   ``assembly_failed``, with a bitmask in ``ToroidalAssemblyDiagnostics``.
7. **Estimate conditioning**, then **solve** the pencil. Failures:
   ``condition_estimate_failed``, ``eigensolver_failed``.

Adding a stage means adding a ``ModeSolveStatus`` value for it. Do not fold a new
failure mode into an existing status; the whole design assumes a caller can tell
which contract broke.

What you may not quietly change
-------------------------------

**The annular model.** The domain runs from ``lambda_inner`` to
``lambda_outer`` and excludes the magnetic axis. The outer surface is a fixed
conducting wall (``xi^lambda = 0``); the inner surface carries the natural
weak-form condition. This is not a detail of the current implementation that a
finer grid would fix. A δW over a truncated annulus with a free inner edge
bounds the full-plasma δW neither from above nor from below, so
``StabilityClassification`` describes the annular model and nothing else. If you
implement magnetic-axis regularity -- which is specific to each harmonic and
each displacement component -- add a new ``RadialBoundaryModel`` value rather
than changing what the existing one means.

**Refusals are load-bearing.** ``n = 0`` and rational-surface topologies are
rejected outright. In both cases the low-level machinery is capable of running:
the DOF layout can already represent all-component one-sided cuts at a resonant
interface. What is missing is the physics that says which admissible space is
the right one. Turning an undecided admissible space into an optimizer-facing
stability number is the specific failure this refusal exists to prevent.

**The Hermitian realification.** Each retained complex coefficient becomes two
real DOFs with ``xi = c cos(...) + s sin(...)`` and ``z = c - i s``, so a
Hermitian entry ``a + i b`` realifies as ``[[a, b], [-b, a]]``. Eigenfunction
reconstruction depends on that sign. Change it in
``toroidal_energy.hpp`` and ``spectral_layout.hpp`` together or not at all.

**Floating-point contraction.** ``src/backend/hip/stability`` is compiled with
``-ffp-contract=off``, matching the equilibrium module. The metric is a chain of
differences whose cancellation makes contraction's effect unpredictable rather
than merely small, and keeping both modules on the same footing means a quantity
can be traced across the boundary without wondering which side reassociated it.

Adding a poloidal harmonic or radial resolution
-----------------------------------------------

``m_max``, ``chebyshev_order``, ``minimum_radial_domains``, and ``n_theta`` are
plain ``StabilityConfig`` fields, but they are not independent. ``n_theta`` must
be at least ``4 * m_max + 1``, enforced in both ``SpectralDofLayout`` and
``validate_config``: products of two harmonics reach ``2 * m_max``, and the
discrete poloidal quadrature has to resolve that without aliasing. Raising
``m_max`` without raising ``n_theta`` is rejected rather than silently aliased.

Cost grows quickly. ``real_order`` is
``2 * 3 * (2*m_max + 1) * (n_domains*chebyshev_order + 1)`` minus the eliminated
outer normal displacements, and the dense eigensolve is cubic in it.

Working on the linear algebra
-----------------------------

The stability backend reaches dense linear algebra only through the public
``quasar::numerics`` headers -- ``generalized_eigensolver.hpp``,
``condition_estimator.hpp``, ``cholesky.hpp``, ``block_tridiagonal.hpp``, and
``shift_invert_lanczos.hpp``. Do not include anything from
``src/backend/hip/numerics/``; that is another module's private implementation,
and if you need a helper from it, promote the helper instead.

Budget the dense operations. A mode solve currently performs a ``syevd`` on the
stiffness, a ``syevd`` on the inertia (that is what
``estimate_symmetric_condition`` is -- an exact spectral condition number, not a
cheap norm estimate), and a ``sygvd`` on the pencil. The assembly's own
positive-definiteness Cholesky is switched **off** by ``StabilitySolver``,
because ``sygvd`` factors the mass term internally and reports the same
condition; the solver maps that back onto the assembly diagnostics so nothing is
lost. Leave ``ToroidalAssemblyConfig::verify_mass_positive_definite`` enabled for
any other consumer.

The shift-invert path
---------------------

``spectral_blocks.hpp`` exposes the fact that the assembled pencil is exactly
block-tridiagonal once reordered radial-major: two DOFs couple only when their
radial nodes share a Chebyshev domain, and the assembly skips non-sharing pairs
outright. Blocks are ``chebyshev_order`` consecutive radial nodes, plus a final
block holding the last shared endpoint -- necessarily non-uniform, because
``n_domains*order + 1`` is not a multiple of ``order``.

Two properties are worth preserving if you touch this:

- The extraction reduces the largest magnitude found **outside** the pattern and
  the solver refuses the path unless it is zero. The structure is a claim about
  the assembly, and the code checks the claim instead of trusting it.
- The path runs *in addition to* the dense solve, and the dense solve still owns
  the classification. Shift-invert returns only the eigenvalues near the shift,
  so it cannot supply ``maximum_absolute_omega_squared``, which is what
  ``eigenvalue_resolution_threshold`` and therefore the stable/unstable decision
  are defined against. Making shift-invert load-bearing requires a
  largest-magnitude estimator for the pencil first. Until that exists, treat the
  path as validated infrastructure rather than as the production solver.

Tests
-----

Unit tests live in ``tests/unit/physics/stability/``. All of them except
``test_spectral_layout`` need a real device and are registered through
``quasar_add_hip_runtime_test``, so a build on a login node stays valid and a run
without a GPU reports a clean skip. They deliberately carry no ``slow`` label:
the default presets exclude that label, and this slice is the feature the tests
exist to cover.

A new numerical stage needs a test that compares it against something
independent -- a closed-form limit, a convergence rate under refinement, or the
dense reference for a small case. A test that compares a stage to its own output
at a different resolution will pass through a sign error in the metric.
