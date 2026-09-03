# Changelog

All notable changes to Quasar are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/), and the project is pre-1.0 so
interfaces may still change between entries.

## [Unreleased]

### Changed
- GPU residency: removed the remaining host-side floating-point *computation*
  from the production paths. An audit found seven places where the CPU still did
  numerical work; six are now kernels and the seventh is documented as belonging
  where it is. The motivation was only secondarily throughput. Five of the seven
  were a *second host definition* of arithmetic that already existed as a kernel,
  kept in agreement by hand-maintained comments — the failure mode the PIC
  particle sampler was ported to eliminate.

  - **MHD reduction second passes** (`src/backend/hip/mhd/mhd_reduce.hip`). The
    CFL max signal rate, both div(B) diagnostics and the positivity admissible
    fraction copied O(N_cells/256) per-block partials back and folded them in a
    host loop, several times per accepted timestep — the only host arithmetic in
    the tree that both scaled with the mesh and ran per step. Now three
    single-block finish kernels. The `device_synchronize` is unchanged and not
    removable: `dt`, `theta` and the div(B) defect gate host control flow. The
    max and min folds are comparison-only and therefore bit-identical; the
    relative-div(B) scaled ratio reproduces the `ScaledValue` fold and its two
    `scalbn` calls exactly, and needs no `-ffp-contract=off` because it contains
    no multiply-add.
  - **MHD background sampling** (`src/physics/mhd/mhd_solver.cpp`). The
    constructor sampled the analytic profile cell by cell on the host with three
    virtual `sample()` calls per padded cell. It now goes through
    `lower_affine_background_profile` and `launch_mhd_sample_background_profile`,
    which gained a cylindrical branch. **Behavioral change:** face coordinates
    move from `origin + i*dx` to `fma(i, dx, origin)`, so a seeded background
    differs in the last bit; this is the coordinate form the solver's own metric
    uses, the same trade recorded in `initial_conditions.hpp`. **Also a
    tightening:** a background profile that is not affine over an element is now
    refused by name at construction rather than sampled pointwise. Both
    registered profiles are affine, so no configuration loses support.
  - **PIC initial field seeding** (`include/quasar/physics/pic/initial_fields.hpp`,
    `src/backend/hip/pic/pic_seed_fields.hip`). All three deck generators
    (`seed_perturbation`, `seed_tm_cavity`, `seed_em_wave`) were NumPy in the CLI,
    pushed through a bare `copy_from_host`. They are now one dispatching kernel
    selected by enumerator, following the `initial_conditions.hpp` model. The
    deck layer keeps parsing, the unit conversion, the validity refusals and the
    O(1) scalars those refusals compute.

    The cylindrical standing mode is the substantive fix: the host code ran an
    *interpreted* per-face loop over a helper that hand-reimplemented the
    axis-even / outer-PEC-odd parity continuation, asserting in a comment that it
    matched the live boundary kernel with nothing enforcing it. The magnetic half
    step is now taken off the **ghost-filled** electric component, so the parity
    is whatever the configured closure actually says it is. The distributed
    runner no longer reaches the seed through a host adapter either.

    **This one changes cylindrical seeded fields by far more than a rounding
    step, and in a good direction.** The host path evaluated `J0` through a
    dependency-free Abramowitz & Stegun polynomial whose documented accuracy is
    ~1.5e-8 absolute; the kernel calls the device `j0`. On an 8-cell radial
    mode the seeded `Ez` moves by **3.1e-8**, and the new value is the correctly
    rounded one — it agrees with a series reference to 1.1e-16, where the old
    value disagreed with it by 3.1e-8. A `seed_perturbation` cylindrical deck is
    therefore now seeded to full double precision rather than to eight digits.
    Any golden for such a deck moves at the 1e-8 scale, and that is the expected
    magnitude, not evidence of a port defect. The zero of `J0` is still located
    host-side (`j0_zero`), because it depends on the mode number rather than on
    position.

    The Cartesian generators move only at ulp scale (device trig and a different
    association in what were `np.multiply.outer` products).
  - **Conserved cell totals** (`src/backend/hip/mhd/mhd_cell_sums.hip`). This
    fixes a real inconsistency: the distributed runtime applied the cylindrical
    `cell_volume` weight and the serial CLI did not, so one axisymmetric deck
    reported two different `mass_initial` values depending on how it was run. One
    compensated device reduction now owns the weight and both paths call it. The
    weighted form is canonical; golden values for cylindrical decks move.
  - **PIC species admissibility** (`src/physics/pic/pic_solver.cpp`).
    `add_species` downloaded all eleven particle planes to look at two of them
    and a flag; it now runs a device predicate.
  - **Equilibrium → MHD projection**
    (`src/backend/hip/equilibrium/gs_mhd_seeding.hip`). `project_to_mhd`,
    `project_fluid` and `max_divergence` were the last grid-scale host loops in a
    module documented as fully GPU-resident. The F and pressure profiles are
    lowered to `ProfileCoefficients` for the usual reason (a vtable cannot cross
    to the device), and the axis-connected plasma mask is taken as an input from
    the existing `launch_gs_build_plasma_mask` rather than recomputed, so there
    is no second flood fill. The host forms remain as the oracle and the
    equivalence test asserts an **equality**, not a tolerance.
  - **Stability q-probe targets** (`src/backend/hip/stability/radial_domains.hip`).
    A small fill kernel replaces a host `linspace` staged through a vector.
    Negligible cost; done for consistency with the Chebyshev node construction
    that already feeds the same trace launch.

  Deliberately **not** moved: `ParticleSpecies::validate_snapshot`. Its data
  arrives on the host (a caller's vectors, an HDF5 checkpoint, a migration
  gather) and must be copied to the device regardless, so validating after the
  upload would remove no transfer, would add a launch and a synchronize, and
  would cost the documented guarantee that a rejected snapshot leaves the
  existing device state untouched — which the distributed seed and the
  checkpoint restore both rely on. The reasoning is recorded at the function.

### Added
- Stability: new fixed-boundary ideal-MHD stability vertical slice under
  `physics/stability` — the framework's first eigenvalue problem, and the first
  end-to-end consumer of a converged `GsDeviceResult`. PEST straight-field-line
  coordinates are built from the traced flux surfaces, rational surfaces are
  located per toroidal mode number, the radial domain is cut at them and
  discretized with Chebyshev–Gauss–Lobatto spectral elements crossed with a
  Fourier poloidal basis, and the compressible Glasser/Bernstein plasma energy
  and inertia are assembled into a dense Hermitian pencil solved by hipSOLVER
  `sygvd`. The energy is assembled in weak form so the operator conditioning
  grows like `order^2` rather than `order^4`. Everything except a compact
  diagnostic summary stays on device. Documented in
  `docs/theory/toroidal_ideal_mhd_energy.rst` and
  `docs/theory/newcomb_cylindrical_energy.rst`.

  The supported model is explicitly annular: the outer surface carries the fixed
  conducting condition `xi^lambda = 0`, while the truncated inner surface gets
  the natural weak-form condition, because magnetic-axis regularity is
  harmonic- and component-specific and is not implemented yet. `n = 0` and any
  rational-surface topology are refused outright rather than approximated. A
  result from this slice must not be presented as a full-axis tokamak result.
- Numerics: deterministic dense GPU linear algebra — a symmetric-definite
  generalized eigensolver, an exact spectral condition estimate, a positive-
  definiteness test, and a pivoted block-Thomas factorization/solve, all in
  hipSOLVER deterministic mode with hipBLAS atomics disabled. hipBLAS and
  hipSOLVER are resolved from the same ROCm prefix the HIP compiler selected, so
  a development host cannot compile against one release and load another.
- Numerics: shift-invert Lanczos (`numerics/shift_invert_lanczos.hpp`) for the
  eigenvalues of a symmetric-definite pencil nearest a shift, iterating in the
  `M` inner product on `(K - sigma M)^-1 M` with full reorthogonalization and a
  reproducible seeded start vector. It reuses one block-Thomas factorization for
  every Lanczos vector, so the per-vector cost is a block back-substitution
  rather than a dense solve.
- Numerics: `BlockPartition` gives the block-tridiagonal solver non-uniform
  block orders. A Chebyshev spectral-element operator has `n_domains*order + 1`
  radial nodes because adjacent domains share their common Lobatto endpoint, and
  that count is never a multiple of `order`, so no uniform blocking of it exists.
  Rectangular off-diagonal blocks avoid the alternatives, which were a padded
  pencil carrying spurious modes or no block structure at all. The uniform
  entry points remain as thin wrappers.
- Stability: `spectral_blocks.hpp` exposes the assembled pencil's exact
  block-tridiagonal structure under a radial-major reordering, and
  `StabilityConfig::eigen_method` can request the shift-invert path on top of it.
  The extraction reduces the largest magnitude found *outside* the
  block-tridiagonal pattern and refuses to proceed unless it is zero, so a
  structural assumption that stops holding is reported rather than silently
  discarding couplings. The dense solve still runs and still owns the
  classification: shift-invert returns only the eigenvalues near the shift and
  cannot bound the top of the spectrum, which is what
  `eigenvalue_resolution_threshold` is defined against.
- Equilibrium: the Grad–Shafranov solver is now fully GPU-resident. Every
  arithmetic stage — the sixth-order compact operator, deterministic reductions,
  Green's-function boundary coupling, source terms and derivative fields, the
  critical-point search, multigrid with defect correction, and flux-surface and
  shape diagnostics — runs through the launch ABI in
  `physics/equilibrium/kernels.hpp`, and `GsSolver` retains only control flow.
  Two consequences are load-bearing: profiles are lowered to a
  `ProfileCoefficients` POD at construction because a vtable cannot cross to the
  device, so only `PolynomialProfile` works today; and the module is compiled
  with `-ffp-contract=off` for both the Padé line solve and the compensated
  current integral.
- Equilibrium: `GsDeviceResult` retains its profile provenance and derived
  fields, so a downstream consumer can rebuild the source terms of the
  equilibrium it was handed without re-deriving them from `psi`.
- Equilibrium: new free-boundary Grad–Shafranov vertical slice — the framework's
  first elliptic boundary-value solver (every prior slice is explicit hyperbolic
  time-marching, and the tree previously contained no Poisson, multigrid, or
  Krylov machinery at all). Sixth-order compact (Padé) operators supply the
  residual, a matrix-free geometric multigrid supplies the correction, and
  defect correction couples them so the converged solution carries sixth-order
  accuracy while every linear solve stays local and smoothable. Free-boundary
  closure uses the exact axisymmetric Green's function (complete elliptic
  integrals via AGM) over external coils plus the plasma's own current.
  Polynomial profiles in normalized flux are renormalized each outer iteration
  to a requested total plasma current. Derived output includes `B`, `q(psi)`,
  flux-surface geometry, and shaping parameters, all computed inside the module
  where the high-order derivatives are still available. Documented in
  `docs/dev-guide/adding_an_equilibrium_profile.rst`.
- Equilibrium: MHD seeding bridge — projects a converged equilibrium onto the
  staggered MHD mesh by differencing `psi` at cell corners rather than
  interpolating `B`, so the result is discretely solenoidal to round-off
  (measured 1.3e-13 relative) and is accepted by
  `MhdSolver2D::seed_background`. This replaces the vacuum Biot–Savart `A_phi`
  projection used by `examples/square_toroid_mhd` with a self-consistent
  equilibrium.
- Numerics: reusable elliptic machinery under `numerics/` — node-centered
  `EllipticGrid`, sixth-order Padé derivative coefficients with derived
  one-sided closures, a pivoting tridiagonal line solve, second- and
  sixth-order Grad–Shafranov operators, matrix-free multigrid, defect
  correction, and a tile-decomposition layer whose tiled operator is
  bit-for-bit identical to the serial one.
- MHD: new ideal-MHD vertical slice — MP5/MP7 monotonicity-preserving
  characteristic reconstruction on Cartesian and axisymmetric cylindrical
  `(r, z)` grids, an HLLD Riemann solver,
  finite-difference constrained transport (FD-CT) for `div(B) = 0`, an SSP-RK3
  integrator, and a conservative troubled-cell positivity control. Driven from a
  new `quasar.mhd` CLI
  (`python -m quasar.mhd.cli run <input.yaml>`) with `_core.mhd` bindings, and
  shipped with the `brio_wu`, `mhd_blast`, `mhd_rotor`, `orszag_tang`, and
  `mhd_linear_wave` example decks. Every scheme axis self-registers and is
  selected by deck string.
- MHD: static background magnetic-field split `B = B0 + b` — runs can carry a
  fixed, discretely divergence-free prescribed field `B0`, including a
  non-uniform current-carrying field, while evolving the perturbation `b`.
  Enabled via a `background_field:` deck block (`enabled`, `profile`, uniform
  `bx0/by0/bz0`, inline `conductors:`, `file:`, or a coil `a_file:`). `B0` comes
  from the pluggable `IMhdBackgroundProfile` registry (built-ins `"uniform"`
  and `"linear_vacuum"`), inline Biot-Savart conductors, or an `.npz` file.
  Setup validates solenoidality with a
  scale-free predicate: the discrete divergence residual is normalized by the
  magnitudes of the directional derivative terms that form it (not by the raw
  field values, so a large DC offset cannot mask a real derivative) and must
  stay within `1024` machine epsilon. A residual at an exact cross-direction
  cancellation is additionally admitted when it is explained by the accumulated
  per-face rounding uncertainty. The physical total field `B0 + b` enters the
  eigensystem, Riemann fluxes, Maxwell stress, and fast-magnetosonic CFL speed
  (so a nonzero `B0` can tighten the stable timestep). The stored split-energy
  variable is internal energy plus kinetic energy plus `0.5|b|^2`; it is not the
  physical total-field energy. After constrained transport supplies `db/dt`, the
  solver applies the exact discrete change of variables
  `dE'/dt = dE_total/dt - B0·db/dt`. `B0` remains fixed, and its discrete
  divergence is validated independently of the CT-preserved divergence of `b`.
  Exposed through the
  `quasar.mhd` CLI/deck, the `_core.mhd` bindings (`MhdBackgroundSpec`,
  `MhdSolver2D.seed_background`/`has_background`,
  `registered_mhd_background_profiles`), and a new `examples/mhd_guide_field`
  guide-field case.
- MHD: inline Biot-Savart `background_field.conductors` in SI decks, with
  solver-derived padded sampling and Cartesian or cylindrical discrete curls,
  so coil-seeded runs no longer require a synchronized coil deck or intermediate
  `A_xyz_grid` file.
- MHD: one-sided non-periodic boundary stencils for the order-2 MUSCL path — at
  `outflow`/`wall` boundaries the `muscl_minmod` boundary-face reconstruction
  uses an interior-biased one-sided slope (dropping dependence on the ghost
  gradient) while still reading the ghost values that impose the wall closure;
  periodic boundaries keep the two-sided wrap stencil unchanged. The `mp5`/`mp7`
  characteristic reconstructions are unaffected: they retain their symmetric
  two-sided stencils at every side and close a non-periodic boundary through the
  filled ghost values alone. The conservative flux difference is unchanged in
  all cases, so discrete conservation still telescopes. The per-side periodic
  classification is owned by the boundary axis
  (`quasar::boundary::mhd_boundary_is_periodic`).
- MHD: the `wall` boundary — a perfectly-conducting wall imposing `n·B = 0` on
  the magnetic field and `v·n = 0` on the fluid, per-side with independent
  fluid/field selection, selectable from an MHD deck via
  `boundary.fluid`/`boundary.field`.
- PIC: axisymmetric cylindrical `(r, z)` mode (`geometry: cylindrical`) — a full
  `m = 0` EM-PIC scheme family (`yee_cyl_o2`, `boris_cyl_*`, `esirkepov_cyl_*`)
  with `1/r` / `(1/r) d(r·)/dr` Yee curls, a finite-volume on-axis closure at
  `r = 0`, ring-volume-weighted charge-conserving deposition, and a cylindrical
  Boris push. Validated by the `cyl_cavity_tm010` (TM010 pillbox resonance,
  dominant FFT peak within a few percent of `j01 c / (2 pi R)` with a clean
  `J0(j01 r/R)` radial profile) and `cyl_gyro_orbit` examples. The axial field
  `Ez` / `Bz` and axial velocity use the `ey` / `by` / `vy` slots; the azimuthal
  `Ephi` / `Bphi` use the `ez` / `bz` slots.
- PIC: `outflow` field boundary kind (`boundary.field: outflow`) — a first-order
  characteristic (Mur) open wall that lets outgoing radiation leave with little
  reflection, at both 2nd- and 4th-order FDTD. A diagonal characteristic update
  owns each outflow/outflow corner, so channels, mixed PEC/outflow boxes, and
  boxes open on all four sides remain stable and absorb outgoing pulses.
- PIC: support for `units: normalized` decks and physically consistent `units: SI`
  decks. SI decks are non-dimensionalized through `Normalization` before stepping
  (grid, dt, charge/mass/velocity/density, external field) and diagnostics are
  converted back to SI on output.
- PIC field boundary conditions are live: per-side `boundary.field` (`periodic` /
  `pec` / `outflow`) selectable from the deck, with stable energy-conserving PEC
  reflection.
- PIC current-smoothing filter pipeline is wired from the deck
  (`numerics.current_filter`: `binomial` / `compensated_binomial`, with passes).
- Field-evaluator selection by registry name: `external_field.evaluator.type` may
  be `biot_savart`, `uniform`, `dipole`, `gradient`, or `file_grid`; the coil CLI
  selects its evaluator through the registry too. `file_grid` loads rectilinear
  NPZ maps in the deck layer (or the portable text format in C++) and provides
  trilinear B plus its exact piecewise Jacobian. Any other live registered
  evaluator can receive generic `params`; finite scalar values normalize to
  one-element vectors and flat finite lists pass through unchanged. Python
  exposes sorted live evaluator-name introspection plus registry and instance
  queries for vector-potential capability.
- PIC `fields.initial` seeding (`seed_perturbation`, `seed_em_wave`, and
  `seed_tm_cavity`) and field-only decks (a deck may define species, an external
  field, or an initial field seed). `seed_tm_cavity` initializes the exact
  order-2/order-4 rectangular-PEC Yee eigenvector, including the magnetic field
  at its leapfrog half-step, so cavity frequency, `div(B)`, wall parity, and the
  discrete energy invariant can be validated without startup-mode contamination.
- PIC species may carry a deterministic sinusoidal
  `initial.velocity_perturbation` (vector amplitude, integer 2-D mode, and
  phase). The two-stream examples now use it with a fixed neutralizing
  background and validate the measured linear growth rate against the cold
  two-beam dispersion relation instead of depending on random particle noise.
- All nine PIC example decks (`two_stream`, `filtered_two_stream`, `landau_damping`,
  `weibel`, `em_wave_propagation`, `beam_in_channel`, `pec_cavity`,
  `magnetized_plasma`, `coil_confinement`) are now runnable, documented, and covered
  by integration tests.
- New `square_quad_field` (Biot–Savart quadrupole field map) and `square_quad_pic`
  (EM-PIC plasma on the quadrupole null) example decks, each with a README and a
  matching `tests/python/test_examples.py` integration test.
- PIC test coverage: a real FDTD plane-wave dispersion check, a behavioral
  mixed-boundary test, Gauss-residual diagnostics tests, an end-to-end CLI run
  smoke test, and a `--log-every` / `--write-every` diagnostics test.
- PIC `quasar pic run --write-every N` writes a distinct, step-indexed,
  self-contained snapshot (`out_<step>.npz`, 10-digit zero-padded step) every N
  steps; the end-of-run aggregate `out.npz` (accumulated snapshot and scalar
  series) is always written as well.
- CPU-only registry-linkage tests for the current-filter and
  field-solver/pusher/deposit registries (a dropped registration now fails in the
  standard suite, not only at device-build time), and coil deck-parse tests for
  the `plane` observation and every geometry type.
- PIC: per-component field-to-host accessors on the solver
  (`field_component_to_host`, `external_field_component_to_host`) so diagnostics
  copy only the requested Yee components from the device instead of the whole
  six-field dict every snapshot.
- PIC/MHD: hybrid MPI and multi-GPU runs selected with `--devices` and
  `--decomposition`, including gathered or sharded diagnostics and
  repartitionable parallel-HDF5 checkpoint/restart.
  * New `quasar.pic.run` and `quasar.mhd.run` entry points accept
    `quasar.distributed.RunOptions` and return `RunResult`, exposing placement,
    diagnostics, checkpoint, restart, and timing policy while plain calls retain
    the serial path.
  * `QUASAR_ENABLE_DISTRIBUTED=AUTO|ON|OFF` controls MPI C++ and parallel-HDF5
    discovery; the Python availability API remains importable in serial-only
    builds.
  * `read_diagnostics_manifest` validates sharded completion records;
    self-contained multi-node launcher examples cover PIC and MHD, and
    collective failures produce a consistent decision on every rank.
  * `hip-gfx942-distributed-{release,debug}` and
    `hip-gfx950-distributed-{release,debug}` presets, plus their `-all` test
    variants, require the distributed dependency set and schedule GPU tests
    through a CTest resource specification.

### Known limitations
- Equilibrium: the Newton phase is implemented but **disabled by default**. Its
  diagonal profile Jacobian is the fixed-boundary term; on a free-boundary
  problem the boundary condition is itself a dense functional of the interior
  solution, and with that block missing the Newton direction is wrong near
  convergence (measured: Picard converges in 222 iterations, Newton stalls at
  7.4e-4). Damped Picard is the production path. Completing Newton requires a
  Jacobian-free Newton–Krylov formulation or the von Hagenow surface-current
  form.
- Equilibrium: the tile layer reports via `TileGrid::supports_local_pade()`
  whether a decomposition can run the sixth-order operator locally; a 2D
  decomposition cannot, because the Padé closures need a pivoting line solve
  spanning the full line.
- Equilibrium: the plasma-contribution boundary integral is the exact
  `O(N_boundary * N_interior)` form. It is negligible at 256² but dominates
  above roughly 1024²; multipole acceleration is deferred.
- Equilibrium: only `PolynomialProfile` is usable. Profiles are lowered to a
  `ProfileCoefficients` POD at construction because the device cannot follow a
  vtable, so a new profile shape has to extend that POD rather than subclass an
  interface. See `docs/dev-guide/adding_an_equilibrium_profile.rst`.
- Stability: the domain is annular and excludes the magnetic axis. The outer
  surface is a fixed conducting wall; the truncated inner surface carries the
  natural weak-form condition. A δW over that truncated annulus bounds the
  full-plasma δW neither from above nor from below, so `unstable` and
  `no_instability_detected` are properties of the annular model, not verdicts
  on the equilibrium. `RadialBoundaryModel` is reported on every result so a
  consumer cannot lose track of which model produced a number.
- Stability: `n = 0` is refused. The cited non-axisymmetric reduction omits an
  additional inductive contribution for that mode.
- Stability: any rational-surface topology is refused
  (`unsupported_rational_topology`). The DOF layout can already represent
  all-component one-sided cuts at a resonant interface, but the continuum
  interface conditions for the two tangential components have not been derived,
  and an undecided admissible space must not become a stability classification.
- Stability: free-boundary work needs wall topology and conserved vacuum flux
  degrees of freedom; the plasma volume form alone is incomplete.
- Stability: the shift-invert path does not yet replace the dense eigensolve.
  It has no way to bound the largest `|omega^2|`, and that bound defines
  `eigenvalue_resolution_threshold` and therefore the stable/unstable decision.
  Skipping the `O(real_order^3)` solve needs a largest-magnitude estimator for
  the pencil as well; until then shift-invert runs alongside the dense solve and
  is cross-checked against it.

### Fixed
- Equilibrium: F-profile recovery and the optional Newton profile Jacobian now
  apply the same plasma-current normalization as the solved source; open flux
  surfaces are excluded from q and geometry aggregates; and projection rejects
  target meshes whose ghost faces reach the cylindrical axis rather than
  emitting a non-solenoidal background.
- Numerics: the pivoting tridiagonal solve handles one-row systems safely, tile
  partitioning rejects non-positive halos, and full-weighting restriction
  validates its exact nested-grid contract before indexing the fine field.
- MHD: split-background momentum fluxes no longer lose material-pressure and
  perturbation-stress gradients beside a dominant static field.  Material,
  background-linear cross, wave-dissipation, and static-background stresses are
  retained as separately scaled pieces through one fused divergence reduction;
  this preserves an order-one gas-pressure force even when `|B0|` is near
  `1e100`, while retaining the same Cartesian and cylindrical balance laws.
- PIC: doubly periodic Maxwell domains now enforce their integral Gauss-law
  compatibility. A nonzero total particle charge is rejected unless the deck
  explicitly selects `neutralizing_background: true`; no implicit ion or fixed
  background is inserted.
- PIC: cylindrical prescribed fields are validated as true axisymmetric 3-D
  evaluators instead of accepting an arbitrary meridional slice. Rotational
  covariance is checked in the configured lab frame, regular axis parity is
  enforced, and magnetic solenoidality is checked from the evaluator's analytic
  gradient. This accepts smooth nonlinear divergence-free fields without
  confusing finite-difference truncation error with a magnetic monopole.
- PIC: post-processing now reconstructs every component on its own Cartesian or
  cylindrical Yee coordinates, keeps independent high faces at non-periodic
  walls, removes only periodic duplicate endpoints, and reads the persisted
  field-boundary topology and sampling plane. Cartesian `plane: xz` plots are
  labelled in the physical `(x,z)` frame with the right-handed component basis
  `(x,z,-y)`, rather than being silently misidentified as `(x,y)`. Plots and
  archived component views therefore exclude ghosts without dropping, shifting,
  or mislabelling physical degrees of freedom.
- PIC and MHD: runs with `t_end` now shorten the last stable step and land on
  the requested end time exactly (within the representable floating-point
  endpoint), rather than overshooting it by one nominal timestep. When `t_end`
  also shortens the very first PIC update, leapfrog magnetic-field seeds use
  that clipped width, so their stored `t=-dt/2` phase matches the actual step.
- PIC: the cylindrical `J0` cavity seed now initializes azimuthal magnetic field
  at the stored `t=-dt/2` leapfrog time using the selected order's exact radial
  Yee curl. The old electric-only initialization led the intended standing mode
  by half a timestep. Radial Bessel indices are supported throughout the
  resolved spectrum and rejected above its Nyquist bound.
- PIC: the shipped Weibel case now carries a deterministic, charge-neutral
  `kx=2*pi/Lx` velocity perturbation. Its physical, ghost-free `Bz` mode follows
  the cold filamentation growth rate instead of relying on quiet-start noise
  that did not produce reproducible growth over the documented run.
- Magnetostatics: ideal-filament observations on a segment now raise a
  singular-field error consistently for `B`, `A`, and `grad(B)` instead of
  returning a finite regularized value or allowing backend-dependent NaNs.
- MHD: corrected the axisymmetric cylindrical `(r, z)` balance. Radial fluid
  fluxes now use ring-area/ring-volume finite-volume weights. The remaining
  pointwise curvature terms are only
  `S_mr = (rho*vphi^2 + p* - Bphi^2)/r` and
  `S_mphi = -(rho*vr*vphi - Br*Bphi)/r`; mass, axial momentum, and energy need
  no separate source in this annular conservative form, and toroidal induction
  remains metric-free.
- MHD: the HLLD seven-wave algebra is now a single host/device-callable core
  (`numerics/hlld_core.hpp`) shared by the device hot path and the host registry
  solver, so the two can no longer drift in their degeneracy guards or
  intermediate-state formulas (they previously used different denominator floors
  and fallback branches).
- MHD: the `mhd_linear_wave` convergence example now emits the seeded `t = 0`
  profile (`state_<name>_initial`) and the example tests compare the final field
  against that exact analytic reference instead of a self-referential harmonic
  fit; the `div(B)` example assertions now read the post-step `divb_linf_final`
  instead of the `t = 0` seed value.
- MHD: corrected Cartesian MP5/MP7 reconstruction to use the finite-volume
  cell-average-to-face coefficients. Point-sample Lagrange coefficients leave an
  `O(dx^2)` defect when applied to evolved cell averages. Matching high-order
  face-to-cell magnetic collocation is used by the EOS and diagnostics. The
  Cartesian spatial residual reaches the selected MP design order on smooth
  data; complete time-dependent convergence remains capped at third order by
  SSP-RK3.
- MHD: the CFL stable-timestep guard now enforces the additive multidimensional
  Courant condition `dt * ((|v_x|+c_f,x)/dx + (|v_y|+c_f,y)/dy) <= cfl`. The
  unsplit residual sums both directional flux differences into one stage update,
  so the previous per-direction `max(...)/min(dx,dy)` bound accepted up to ~2x the
  stable step on an isotropic grid (an `auto`-`dt` run with `cfl` near 1 could go
  unstable).
- MHD: the axisymmetric geometric source is now applied at the on-axis column
  `i = 0` (cell-center radius `r = 0.5*dr > 0`); the previous `r <= 0.5*dr` guard
  wrongly skipped that whole column, dropping a nonzero `1/r` source while the
  Cartesian flux difference still ran — breaking conservation at the axis for any
  state with radial flow.
- PIC: `total_em_energy` and `gauss_residual` diagnostics now use the
  axisymmetric metric for cylindrical `(r, z)` runs — the field energy integrates
  over the ring volume `2 pi r dr dz` (`Grid2D::cell_volume`) instead of a flat
  `dx dy`, and the Gauss residual uses the `(1/r) d(r E_r)/dr + d E_z/dz`
  divergence with the natural `r = 0` closure. Both previously reported
  physically wrong values for a cylindrical run (axis cells over-weighted,
  large-`r` cells under-weighted).
- PIC: the explicit-`dt` CFL guard in `EmPic2D3V` now keys off the scheme that
  actually runs. Cylindrical geometry supports both 2nd- and 4th-order curls and
  uses the matching order-dependent limit; `step()` rejects every over-CFL
  (unstable) step, including a shortened variable-width final step.
- PIC: the `axis` boundary name is now rejected on any side other than `x_lo`
  (the `r = 0` inner radius) at deck validation instead of silently doing nothing.
- PIC: quiet-start macro-particle weighting no longer biases the initial number
  density when `n_particles` is not a perfect square. A rank-1 lattice places
  exactly `N` centred, stratified samples per axis, and every Cartesian particle
  carries `density * block_area / N`; cylindrical starts use equal ring-volume
  strata and `density * block_volume / N`.
- PIC: the particle field gather now honors per-axis boundaries. It previously
  wrapped the interpolation stencil periodically on every axis; on a non-periodic
  (wall) axis it now clamps into the boundary ghost layer (matching the deposit),
  so near-wall particles see the boundary field instead of the wrapped far edge.
  The sampled external field's ghost layer is edge-replicated to match.
- Coil: `observation.plane` now normalizes the `u_axis_xyz` / `v_axis_xyz`
  direction vectors before scaling by the extent, so a non-unit axis no longer
  silently changes the sampled plane size; a zero axis raises a clear error.
- PIC: charge conservation across particle boundary crossings. The periodic-wrap
  kernel co-shifts the previous particle position, and specular reflection now
  deposits into ghost cells with stagger-aware image-charge/current fold-back
  instead of mirroring the previous position outside the domain (which had
  teleported current to the far edge via the periodic wrap). Cylindrical image
  charge uses ring-volume ratios. Absorbing particles deposit the complete
  finite-shape loss through the wall before they are killed, and the upper normal
  electric face advances with that boundary current, preserving Gauss's law.
- PIC: the FDTD curls are adjoint (forward E-update / backward B-update). With both
  curls forward, a hard field wall was exponentially unstable; the adjoint form is
  the standard Yee scheme and leaves periodic results unchanged.
- Magnetostatics: `helix`/`solenoid` vertex count is computed in `size_t` with an
  upper bound, fixing signed-int overflow on large `n_turns * n_segments_per_turn`.
- PIC: external fields are sampled component-by-component at the same Yee
  sub-lattice used by the internal fields and stagger-aware gather, eliminating
  half-cell force and magnetic-rotation phase errors.
- `quasar.pic.cli` persists the Yee ghost width (`nghost`) into `out.npz`, and
  `quasar.pic.postprocess.reshape_with_ghost` strips the halo using that explicit
  value instead of reverse-engineering it from the flat buffer size.
- Registry: type-keyed factory registration. Stateless `make_unique` lambdas were
  collapsed by identical-code folding, so `Registry::create(name)` could return
  the wrong concrete type — the root cause of the long-standing PIC field/particle
  boundary "heisenbug". Boundary dispatch now goes through the interface.
- PIC: the periodic particle wrap is side/axis-aware, so periodic and
  non-periodic (e.g. absorbing) walls can coexist per side.
- PIC: current filters now honor boundary periodicity instead of always wrapping
  across the far edge; non-periodic axes clamp the filter stencil at the wall.
- PIC: oversized particle displacements in the Esirkepov deposit now fail loudly
  instead of silently truncating the current-deposition window.
- PIC: `dipole` and `gradient` external-field deck evaluators now require and use
  their evaluator parameters rather than default-constructing zero/trivial fields.
- PIC: `PicDeck.validate()` now rejects non-finite (`NaN`/`inf`) and out-of-range
  deck values up front — domain lengths/origin, normalization reference density,
  `dt_s`, species charge/mass/density/temperature/drift, external-field vectors,
  initial-field amplitude, diagnostics cadence, and unknown `diagnostics.fields`
  names — instead of letting them propagate into the solver. `diagnostics.fields`
  names are normalized to lowercase on every construction path, and
  `--steps-override` only accepts a positive integer.
- Coil: `observation.points` decks are now bounded by `MAX_OBSERVATION_POINTS`
  like the grid/plane/line observation kinds.

### Changed
- MHD: **the seeded initial state is built on the device.** The six benchmark
  generators behind `initial.type` were NumPy expressions over the padded grid
  in `quasar.mhd.io`; they are now one kernel dispatching on an enumerator
  (`physics/mhd/initial_conditions.hpp`), with the deck layer reduced to parsing,
  validation and an O(1) scalar parameter block. The deck still selects a
  generator by string, and `_core.mhd.registered_initial_conditions()` is the
  list the validator checks against.

  The primitive-to-conserved assembly, the face-to-cell magnetic recollocation
  and the positivity preflight moved with them. The recollocation now calls the
  solver's own `mhd_staggering.hpp` quadrature instead of a NumPy mirror of it,
  so the seeded energy and the EOS read the same B by construction rather than
  by two implementations agreeing.

  One behavioural change to expect: cell coordinates come from
  `Grid2D::x_at_cell_center`/`y_at_cell_center`, which are FMAs, where the NumPy
  used `origin + (i + 0.5) * d`. Seeded states move by an ulp. This is the
  better of the two — it is the coordinate mapping the solver already uses for
  every geometric factor, so the seed is now consistent with the mesh instead of
  merely close to it. No example reference needed re-baselining.
- MHD: **the static background field B0 is built and proved on the device.**
  All five deck sources — uniform, named analytic profile, explicit `file`,
  `a_file` corner potential, and inline `conductors` — assemble B0 in device
  memory and run the WP2 solenoidality and boundary-closure sweeps there before
  anything is downloaded. The `conductors` path no longer round-trips: the
  padded corner grid expands into a device point cloud, the Biot–Savart
  evaluator writes device planes, and the lab-Y plane feeds the curl directly.
  Reading an npz is still host work, because loading a file is not calculation.

  Analytic profiles are lowered to an affine POD, because a vtable cannot cross
  to the device. The lowering is derived by probing the registered profile
  rather than by re-listing the profiles, so the registry stays the single
  definition of what a name means — and a profile that is not affine is refused
  by name instead of being silently linearized. Both built-ins are affine, which
  is exactly the condition `IMhdBackgroundProfile::sample` already requires for
  its returned value to be the element's finite-volume moment.

  The opt-in cylindrical vacuum projection is the same preconditioned conjugate
  gradient as before, on the device: same operator, same Jacobi preconditioner,
  same `5e-11` relative stopping contract against the same characteristic
  field-derivative scale. Its inner products now use the deterministic
  double-double tree, so the result is bitwise reproducible across block counts,
  which NumPy's pairwise sum was not obliged to be. One deliberate one-ulp
  difference: the boundary ring is Dirichlet data and is returned exactly as
  supplied, where the NumPy divided the whole array back through `psi = r*A`.

  The plan for this work called for reusing the equilibrium defect-corrected
  multigrid here, on the correct observation that the projection operator is
  `Delta*` up to a positive row scaling by `r`. That is not taken:
  `GsDeviceMultigrid` is declared in `physics/equilibrium/kernels.hpp`, so
  calling it would create an mhd -> equilibrium physics edge. Its host twin
  already lives on the numerics axis, so the right fix is to move the device
  class down to numerics as well — a refactor that touches equilibrium and
  stability and does not belong here. See the note at the top of
  `src/backend/hip/mhd/mhd_background_build.hip`.
- MHD: `MhdSolver2D::state_component_to_host` collocates `bx`/`by` to cells on
  the device before the download rather than after it. The bytes cross the bus
  either way; the last per-cell arithmetic in the solver no longer runs on the
  host.

- Distributed PIC: **particle migration routing runs on the device.** Deciding
  where a departing particle goes is arithmetic — a periodic coordinate is
  wrapped back into the global domain, divided by the cell width and floored to
  a global cell index, and that index is mapped through the balanced tile
  partition to an owning endpoint — and all of it used to run on the host, once
  per departing particle, over a snapshot downloaded for the purpose.
  `extract_pic_departing_particles_device` now keeps the departing block where
  the partition kernel wrote it, and `launch_pic_route_departing_particles`
  wraps, resolves the owner and looks up the destination rank in one launch.
  What crosses to the host is a routed array of wire records that the host
  slices and memcpys but never inspects field by field.

  Ordering moved with it. Records come back grouped by destination rank and, in
  each group, sorted by ascending stable id; arrivals are re-sorted by id as
  part of the append. A species' particle order therefore no longer depends on
  the order the partition kernel emitted departures or the order MPI delivered
  peers. The sort is a stable counting sort applied over the eight bytes of the
  id and once more over the rank, and it also replaces the host
  `unordered_set` behind the checkpoint id-uniqueness check.

  Still host arithmetic, deliberately: the seed and checkpoint-restore routing
  in `seed_local_species`, `replace_local_species_particles` and
  `route_checkpoint_species`. Those route particles that originate on the host,
  they run once per run rather than once per step, and they route a dead
  particle by id rather than by position.
- PIC: **initial particle sampling runs on the device, and seeded velocity
  draws have changed.** The quiet-start lattice, the Maxwellian sample, the
  optional sinusoidal perturbation, the `|v| < c` check and the in-domain check
  were NumPy in `quasar.pic.cli`, evaluated per particle on the host and then
  uploaded. They are kernels now, writing straight into the species' device
  planes through `EmPic2D3V::sample_species_particles`. What stays on the host
  is O(1) configuration arithmetic: the lattice stride, the per-particle block
  measure, the thermal speed.

  The generator changed with the move. `numpy.random.default_rng` is a stateful
  PCG64 stream, and a stream cannot be evaluated in parallel at an arbitrary
  index without advancing through it; the replacement is Philox4x32-10 keyed on
  `(seed, species)` and *counted* by particle index, so a particle's draw is a
  pure function of where it sits. That makes the sample independent of block
  size, of thread scheduling, and of how the population is partitioned across
  devices. The antithetic pairing is unchanged, so the thermal sample still
  carries exactly zero bulk momentum. **Seeded PIC decks draw different
  velocities than they did before.**

  One consequence was a latent inconsistency, now fixed: the distributed runner
  carried its own host Philox sampler alongside the CLI's NumPy one, so a
  serial and a multi-rank run of the same deck drew different velocities. Both
  now drive the same kernels, and `test_pic_distributed_runner` pins that they
  agree exactly.

  `quasar/pic/initial_conditions.py` is gone. Its only consumer was the CLI, and
  leaving a second host sampler in the tree would have meant a user script and
  the CLI producing different physics from one deck. Its tests were rewritten
  against the sampler that actually runs.
- Examples: `landau_damping` now uses 65536 particles and a `4e-3` perturbation
  instead of 8192 and `1e-3`, and its integration test measures the first four
  peaks of the damped sequence rather than peaks inside a fixed `[3, 11]`
  window. The old form was tuned to one sampling realization: it excluded the
  excitation peak, included the late finite-N recurrence where particle noise
  pushes the mode back up, and passed because that recurrence happened to land
  outside the window for that one draw. At 8192 particles the `k = 1` mode sat
  close enough to the thermal noise floor that the ensemble envelope showed no
  net damping at all. The new particle count keeps the mode above the floor for
  all four peaks in every seed tried, the peak times now match the analytic
  quarter-period and half-period spacing so no window has to be chosen, and the
  rate tolerance is the measured seed-to-seed spread rather than a guess.
- Numerics/magnetostatics/analytic fields: **`IFieldEvaluator` is now
  device-resident, and this is a breaking interface change.** `evaluate_B`,
  `evaluate_E`, `evaluate_A` and `evaluate_grad_B` take a
  `core::DevicePointCloud` and return `core::DeviceVectorField` (three SoA
  component planes) or `core::DeviceTensorField` (nine component-major planes)
  instead of host `Field<Vec3>` / `Field<Mat3x3>`. Callers that genuinely need
  host values — the `quasar.coil` CLI, the Python bindings, tests — call
  `.to_host()` at that boundary; callers that feed another device stage no
  longer round-trip at all.

  Biot–Savart got shorter rather than longer: it already computed into device
  SoA planes and then transposed them to host AoS purely to satisfy the old
  signature, and that staging is deleted. The four analytic evaluators
  (`uniform`, `gradient`, `dipole`, `file_grid`) had no device path whatsoever
  and are now kernels under the new `src/backend/hip/analytic_fields/` module,
  reached through the launch ABI in
  `include/quasar/physics/analytic_fields/kernels.hpp`. `FileGridEvaluator`
  keeps its node values on the device from `configure()` onward, and its
  O(nodes) finiteness and trilinear-solenoidality admissibility sweeps run there
  too; only the scalar descriptor checks stayed on the host.

  Since a kernel cannot throw, each reports failure by OR-ing bits into an `int`
  status word with an integer atomic — exact and order-independent, so the
  reported status does not depend on the launch geometry — which
  `core::throw_on_evaluator_status` turns back into the same exception types the
  host code raised.

  The host evaluators carried their intermediates in `long double`, relying on
  its wider exponent to form products like
  `moment_scale * mu0_over_4pi * inv_r^3` without overflowing on the way to a
  representable answer. A device has no `long double`, so the extended range is
  now carried explicitly: the physics-neutral scaled-arithmetic toolkit was
  factored out of `numerics/mhd_state.hpp` into
  `include/quasar/numerics/scaled_arithmetic.hpp`, and the analytic kernels
  reduce through its exact expansions. That is strictly better than what it
  replaces for the cancelling cases — the dipole's zero-`B_z` cone, a gradient
  field whose offset cancels its displacement term — where the old code used a
  hand-rolled largest-exponent-pair merge. As everywhere else in this port, the
  new module is compiled `-ffp-contract=off`; contraction silently degrades the
  two-sum to a naive sum.

  Verified by `tests/unit/physics/analytic_fields/test_analytic_device_accuracy.cpp`,
  which builds its own `long double` oracle, asserts the device is no worse than
  a naive `double` evaluation of the same closed form where that formula
  cancels, and asserts bitwise reproducibility across repeated launches.

  `pic::sample_external_field` follows the evaluator onto the device. It used to
  build the Yee sample coordinates, validate the evaluator's answers, and map
  them into solver units in host loops over the full padded grid, six components
  deep, and up to three times per component for the cylindrical covariance
  probes. All of that is now kernels in
  `src/backend/hip/pic/external_field_hip.hip`: point construction, the rotated
  covariance probes, the trace(grad B) solenoidality check, the component
  mapping, and the radial-parity verify-and-enforce pass. The prescribed field
  is published device-to-device and never touches host memory. The host keeps
  the scalar tolerance configuration and the choice of which exception to raise,
  so the messages are unchanged with one exception: the solenoidality failure no
  longer names the offending sample index, because an OR-reduced predicate does
  not carry one and recovering it would mean a second pass purely for
  diagnostics.

  One deliberate non-change: `BiotSavartEvaluatorF`, the fp32 sibling, is not on
  `IFieldEvaluator` and keeps its host `Field<Vec3f>` returns. Its only consumer
  is the precision-comparison test, which is an output boundary; a float
  `DeviceVectorField` for one test caller would be abstraction without a second
  user.
- Numerics: **the cylindrical radial moment tables are solved on the device**,
  in one batched factorization rather than thousands of host eliminations.

  `RadialTables` builds roughly twenty stencil rows per radial index across the
  padded grid. Each is an independent general system of width at most eight --
  a moment matrix against monomial data, or a Vandermonde matrix against moment
  data -- and the host solved them one at a time with its own partial-pivoted
  Gaussian elimination in `long double`. The assembly, the iterative-refinement
  defect, the normalization and the residual are now kernels, and the two
  eliminations are one batched `rocsolver_dgetrf_strided_batched` /
  `dgetrs_strided_batched` each, behind a new public
  `numerics/batched_lu.hpp`.

  rocSOLVER rather than hipSOLVER because hipSOLVER exposes only single-matrix
  `getrf`/`getrs`; driving that once per 8x8 system would spend all its time on
  launch overhead. `rocblas`/`rocsolver` join the pinned ROCm package set in
  `cmake/QuasarLinearAlgebra.cmake`, resolved from the same compiler root as
  hipBLAS and hipSOLVER. Determinism needs no flag here: each system is factored
  independently, so there is no cross-system reduction whose order could vary.

  The precision question was the real one, and it was measured rather than
  assumed. Dropping from `long double` to binary64 on an ill-conditioned
  Vandermonde system could plausibly have stopped satisfying the 1e-11
  acceptance threshold `RadialTables` enforces on every row. It does not: swept
  over the padded grid for every width and measure the tables build, the worst
  binary64 residual is 3.4e-12 against a 2.0e-12 `long double` reference. The
  iterative-refinement step is what buys that margin -- it makes the result
  backward stable, so the residual tracks working precision times the matrix
  norm rather than the condition number. A new equivalence test carries its own
  `long double` oracle and asserts all of this, plus that batching changes no
  row and that repeated solves are bitwise identical.

  One measurable regression, reported rather than hidden: the R6 v_lc
  extrapolation slope factors are now one ulp from the exact small rationals
  (1/4, 25/44, 19/30, 7/8) the `long double` path produced. The difference is in
  the moment integral's binomial expansion, not in the division -- reformulating
  to a single quotient does not recover it -- and it is inert in what the value
  is for, an MP limiter bound. The interpolation coefficients in the same rows
  are unaffected and still assert exactly; only those four assertions were
  relaxed to a four-ulp bound.

  **Breaking:** `normalized_cell_moment`, `solve_radial_row` and
  `radial_gauss_weights` take and return `Real` rather than `long double`, and
  `RadialCellMeasure` moved to a new `numerics/radial_cell_moments.hpp` that
  carries the `QUASAR_HOST_DEVICE` closed forms shared by the host accessors and
  the kernels. The batched `solve_radial_rows`, `radial_gauss_weight_rows` and
  `radial_face_extrapolation_factors` are the primary interface; the single-row
  forms are one-element batches. `src/numerics/radial_moments.cpp` is deleted.
- Core/magnetostatics: **coil geometry and observation-point generation moved to
  the device**, which removes the last per-point host arithmetic in front of the
  now device-resident field evaluators.

  A structured observation set is a description, not data: `ObservationGrid`,
  `PlaneSlice` and `LineProbe` expand to one point per lattice site through a
  short affine formula. They gain `to_device_point_cloud()`, backed by kernels in
  the new `src/backend/hip/core/`, and the deck loader and the Python bindings
  use it, so the coordinates are generated where they are consumed. The
  expansion agrees with the host `point_at()` accessors **bit for bit** — which
  is why the line-probe kernel transcribes the C++20 `std::lerp` algorithm rather
  than the obvious `a + t * (b - a)`. `std::lerp` is specified to be exact at
  both endpoints and monotonic; the naive form is neither, and a probe whose last
  point drifted off `end` by an ulp would silently disagree with the accessor a
  deck echoes. That module is compiled `-ffp-contract=off` to keep the guarantee.

  Filament vertices now live on the device too. The generators write them with a
  kernel in `src/backend/hip/magnetostatics/geometry_hip.hip`, and
  `ConductorSystem` flattens them into per-segment planes with another and caches
  the result, so `evaluate_B` has nothing left to upload — both operands are
  already resident. The generator kernels are not one line each because they
  carry the resolvability checks the host loops did: a displacement requested at
  a large centre can lose an entire local dimension to rounding while the
  surviving coordinates still form non-zero segments, so every material
  component of every vertex's offset must survive both the scaling and the
  translation with at least one useful binary digit.

  The angle is rebuilt from the integer index at every vertex rather than
  accumulated, and for a helix the index is reduced modulo the segments per turn
  first. That is what keeps a 400-turn coil free of argument-reduction drift, and
  there is now a test asserting the last turn's transverse coordinates are
  bit-identical to the first's.

  `BiotSavartEvaluatorF` keeps its host `Field<Vec3f>` results, but its fp32
  narrowing is a kernel now: both operands arrive as fp64 device planes and the
  shared origin is subtracted before the cast, so a rigid translation stays
  invisible to the narrowing.

  **Breaking:** `Filament::points` is a `FilamentPoints` device SoA rather than a
  `std::vector<Vec3>`, which makes `Filament` and `ConductorSystem` move-only
  and `ConductorSystem` copy-construction an explicit device allocation.
  `ConductorSystem::segments_soa()` is replaced by `device_segments()`;
  `to_segments_soa()` remains as its host staging view. In Python,
  `Filament.points` is a read-only property that downloads, `Filament.n_points`
  reports the count without downloading, and `ConductorSystem.add` moves from its
  argument. `generic_polyline` is unchanged: its coordinates come from a deck
  rather than a calculation, so they are validated on the host and uploaded.

  What deliberately stayed on the host is scalar and does not scale with the
  point count: the orthonormal basis built once per generator call from one axis
  vector, a racetrack's four frame points and two corners, and the linear
  independence check on a plane slice's two step vectors.
- PIC: the diagnostics no longer round-trip through the host. `alive_count` was
  already a device reduction; `total_kinetic_energy`, `total_em_energy` and
  `gauss_residual` each used to download all six field components and every
  particle record and reduce them in `long double` on the CPU. They are now
  device reductions (`src/backend/hip/pic/diagnostics_hip.hip`), as is the
  one-time neutralizing-background charge sum in
  `EmPic2D3V::initialize_neutralizing_background` and its distributed
  counterpart. What remains on the host is a three-scalar epilogue that
  normalizes the result and chooses which exception to raise; its cost no
  longer depends on the grid or particle count.

  The reductions follow the standard set by the Grad–Shafranov port: a
  deterministic double-double (Knuth two-sum) tree with no floating-point
  atomics, now factored into `src/backend/hip/reduction_detail.hpp` and shared
  with `gs_reduce.hip`. Because the summed terms span an enormous dynamic range
  — an electron mass times a macroparticle weight times a cell volume — each
  term is carried as a mantissa and a binary exponent rather than being
  multiplied out, which is the device form of the host `ScaledPositiveSum` that
  was deleted (`src/backend/hip/scaled_reduction_detail.hpp`).

  This is not bit-for-bit with the deleted host code, and cannot be: `long
  double` has no device equivalent and a parallel tree does not sum in host
  order. Accuracy is instead pinned by
  `tests/unit/physics/pic/test_diagnostics_reduction_accuracy.cpp`, which builds
  its own long-double oracle and requires the device result to be within a few
  ulps of it, to be at least an order of magnitude better than a naive double
  sum, and to be bitwise reproducible across launches. Compensated accumulation
  in double is worth roughly 106 bits of effective mantissa, so the narrower
  base type is not a downgrade against the old long-double accumulator.

  `diagnostics_hip.hip` is compiled `-ffp-contract=off`. This is load-bearing:
  contraction does not perturb the two-sum slightly, it silently turns the
  compensated sum back into a naive one.
- MHD: prescribed-background validation runs on device. Checking that a supplied
  B0 is discretely divergence-free, compatible with the configured homogeneous
  closure, and (when asserted) discretely curl-free used to download all three
  padded components and sweep the grid on the host — including in
  `ensure_background_solenoidal()`, which did it lazily on every first use. The
  sweeps are now `launch_mhd_validate_background_{solenoidal,boundaries,curl_free}`
  (`src/backend/hip/mhd/mhd_background_validate.hip`); the host keeps the
  comparison against the tolerance and the throw.

  The scaled-arithmetic metrics these checks are defined in moved to
  `include/quasar/numerics/mhd_background_metrics.hpp` as `QUASAR_HOST_DEVICE`.
  They had already been duplicated once — `mhd_reduce.hip` carried its own device
  copies for the live div(B) diagnostic — so there were two definitions of the
  same discrete solenoidality criterion that could drift. There is now one.
  (`residual_is_roundoff_explained` deliberately remains separate in
  `mhd_reduce.hip`: its six-argument form asks a different question, namely
  whether the solver owns the state it is judging.)

  Ordering change worth knowing: the constructor now uploads B0 and then
  validates the device buffers, where it previously validated the host vectors
  first. The sweep therefore reads exactly the bytes the solver will use.

  These are equality ports, not accuracy-ordering ones — every reduction here is
  a maximum, which is associative and rounds nothing. The kernels reproduce
  `std::max`'s NaN behaviour rather than `fmax`'s, so a NaN defect is dropped and
  an infinite one propagates exactly as before. Boolean outcomes use integer
  atomics, which are exact and order-independent; the no-floating-point-atomics
  rule is unaffected. `mhd_background_validate.hip` is compiled
  `-ffp-contract=off` because the check compares a residual against the
  representation roundoff of its own operands, and contraction moves those two
  by different amounts.
- Testing: long-running tests are now labelled `slow` and excluded from the
  default `ctest` presets, so `ctest --preset <name>` completes in minutes
  instead of over an hour. Only `python_test_examples` currently carries the
  label — it drives every example CLI end-to-end. New `<name>-all` test presets
  run the complete suite including labelled tests, and `ctest -L slow` selects
  just them. Run the `-all` preset before a release and after changing
  `examples/`, a physics CLI, or the deck schema.
- MHD: the ideal-MHD solver hot path now runs entirely on device (HIP, gfx942).
  Previously several pieces staged device buffers back to the host each step and
  computed there; those are now device kernels:
  * High-order Cartesian flux reconstruction — the characteristic
    monotonicity-preserving
    MP5 (5th-order) and MP7 (7th-order) schemes now run in the device
    reconstruction kernel, and the device path honors the selected scheme order.
    Previously the device kernel silently ran 2nd-order MUSCL-minmod for *all*
    orders, so `mp5`/`mp7` decks were not actually high-order on device; their
    smooth Cartesian spatial residual now reaches the selected MP design order
    (the full method remains third-order in time under SSPRK3). This required
    making the MHD characteristic
    eigensystem and characteristic projection device-callable
    (`QUASAR_HOST_DEVICE`) and extracting the scalar Suresh–Huynh MP limiter
    helpers into a shared `QUASAR_HOST_DEVICE` header
    (`include/quasar/numerics/mp_limiter.hpp`).
  * MHD boundary ghost fills (fluid and magnetic-field; `periodic` / `outflow` /
    `wall`) now run as device kernels instead of host-staged fills.
  * The positivity controller (`troubled_cell`) computes per-cell convex
    density/internal-energy bounds on device. An inadmissible SSP-RK stage is
    rolled back and conservatively CFL-subcycled with a piecewise-constant HLL
    anchor, avoiding floor-driven mass/energy injection. The old device floor is
    retained only as an explicit standalone repair API.
  * The CFL stable-timestep scan and the `div(B)` L-infinity diagnostic now run
    as device reductions instead of staging the whole field to the host.
  The registry-facing scheme/boundary classes are now thin launchers over the
  per-physics device launch ABI; behavior is selected by deck string exactly as
  before (registry names unchanged: `muscl_minmod` / `mp5` / `mp7`,
  `troubled_cell`, `fd_ct_christlieb`, `periodic` / `outflow` / `wall`).
- MHD: the device characteristic MP reconstruction falls back to 2nd-order MUSCL
  for any single interface whose high-order reconstructed state is non-finite or
  non-positive (a per-interface positivity guard), so a few troubled interfaces
  cannot poison the whole sweep.
- MHD: axisymmetric constrained transport and radial fluid fluxes now use exact
  annular face/volume weights, so the discrete cylindrical divergence
  ``(1/r)d(r B_r)/dr + dB_z/dz`` telescopes with the CT curl. Finite-inner-radius
  annular domains and total-field cylindrical curvature sources (including a
  static ``B0``) are supported.
- MHD: cylindrical MP5/MP7 now use equation-matched finite-volume moments
  along the radial direction while retaining Cartesian coefficients axially:
  annular `r dr` rows for mass-like variables, `r^2 dr` rows for azimuthal
  momentum, and unweighted `dr` rows for the toroidal field. High-order
  reconstruction, point recovery, transverse flux quadrature, magnetic
  collocation, and constrained transport are supported on both annular domains
  and grids that include the `r=0` axis; native and Python deck validation now
  accept `mp5` and `mp7` for cylindrical geometry.
- MHD: cylindrical `muscl_minmod` now uses the radius-weighted four-point
  face-to-cell, cell-to-face, and radial corner-EMF rows.  This intentionally
  changes order-2 cylindrical results from the former two-point Cartesian
  collocation while leaving Cartesian MUSCL unchanged.
- PIC field boundaries are now imposed by one-sided / characteristic node
  corrections after each curl rather than by filling a ghost halo. `pec` keeps
  its reflecting, energy-conserving behavior (tangential E / normal B pinned;
  remaining components closed with one-sided stencils) at both FDTD orders; the
  4th-order boundary closure is locally reduced to 2nd order on the outer two
  layers (interior 4th-order convergence is preserved). Periodic still uses the
  ghost wrap.
- The kernel-launch ABIs moved to installed public headers; the
  physics/boundary/numerics modules no longer reach into the private
  `src/backend/hip/` tree (the blanket `src/` include path was removed from the
  module CMake helper and re-granted only to the backend HIP modules). The PIC
  ABI now lives at `include/quasar/physics/pic/kernels.hpp` (a per-physics seam)
  so the backend axis stays physics-neutral; the magnetostatics raw-pointer ABI
  likewise lives at `include/quasar/physics/magnetostatics/kernels.hpp`.
- The numerics `IFieldEvaluator` takes an axis-neutral `core::IFieldSource`
  (implemented by `magnetostatics::ConductorSystem`) instead of naming the
  magnetostatics type directly, so the analytic-field evaluators no longer depend
  on the magnetostatics module. The Python `IFieldEvaluator.evaluate_B` /
  `evaluate_grad_B` binding accepts any `IFieldSource` polymorphically rather than
  the concrete `ConductorSystem`.
- `IFieldEvaluator` gains a `configure(params)` seam (a name→flat-float-list
  parameter map) applied after registry construction, so the PIC driver builds
  every external-field evaluator purely by registry name and configures it from
  the deck — the previous per-type `if/elif` ladder in `quasar.pic.cli` that
  hand-picked a constructor is gone, matching the registry-only coil CLI.
- Registry registration now rejects empty names, empty factories, and duplicate
  names with `std::invalid_argument`. A duplicate can no longer silently replace
  the original factory, and registry registration, lookup, and introspection are
  synchronized for concurrent callers.
- Backend headers under `include/quasar/backend/` are now HIP-free: an opaque
  `stream_t` and backend-neutral device-memory free functions, with all HIP calls
  confined to `src/backend/hip/`. The kernel-launch ABIs no longer leak
  `<hip/hip_runtime.h>` into the physics/boundary/numerics layers.
- Observation-point types (`PointCloud`, `ObservationGrid`, `PlaneSlice`,
  `LineProbe`) moved to `quasar::core`; the numerics `IFieldEvaluator` interface
  no longer depends on a magnetostatics header.
- `BiotSavartConfig` drops the never-consumed `tile_segments` / `block_size`
  fields; kernel tiling is compile-time and tuned per-gfx in
  `cmake/QuasarLaunchParams.cmake`.
- The `quasar-coil` and `quasar-pic` CLIs are standardized: both use the
  `command` subparser dest, a `func` handler, and a default-quiet output model
  with a `--verbose` flag. The coil CLI's previous default-chatty `--quiet`
  behavior is replaced by `--verbose` (off by default).
- PIC `EmPicConfig` carries the particle shape as a deck-vocabulary string
  (`shape`: `cic` / `tsc`) instead of an integer `shape_order`; the solver derives
  the field-solver / pusher / deposit registry names directly from the deck order
  and shape, dropping the internal switch ladders. The specular current fold-back
  is now a `fold_current` hook on `IParticleBoundary` rather than a special case
  in the solver step.

### Removed
- MHD: `python/quasar/mhd/numerics.py` is gone. It was the host mirror of the
  face-to-cell collocation, the EOS, and the background solenoidality/boundary
  sweeps; every one of those now runs on the device, and the standing rule for
  this port is that displaced host code is deleted rather than kept as an
  oracle. The three tests that consumed it either moved to the device path
  (`build_background_from_arrays` for the boundary-closure rules) or carry a
  small, explicitly test-local reference. `tests/python/test_mhd_numerics.py`
  went with it: it tested the mirror against the C++ goldens, and
  `tests/unit/numerics/test_mhd_cylindrical_collocation.cpp` already proves the
  surviving side against polynomial exactness directly.
- MHD: the `reflecting` boundary name has been removed; it is renamed to `wall`
  (same perfectly-conducting semantics), so decks must now select `wall` and
  `reflecting` is rejected at validation.

### Optimized
- MHD: the constrained-transport cell EMF no longer re-derives its tensor
  quadrature per Gauss node. The point reconstruction at cell node `(qx, qy)`
  factors into an x-pass whose result does not depend on `qy`, so those rows are
  built once per x-node instead of once per node pair, and the cell-centred
  `Bx`/`By` stencil block (each entry its own multi-tap face-to-cell reduction)
  is materialised once per cell rather than re-derived for every x-node. That
  block is further shared across a thread block through an LDS tile, since
  neighbouring cells overlap in all but one row of it. Reduction order is
  unchanged throughout, so the EMF is bit-for-bit what it was.
- MHD: the CT corner EMF evaluates each directional face EMF once into a new
  per-face table (`EmfField2D::xface_ez` / `yface_ez`) instead of re-solving the
  same face Riemann problem for every corner whose interpolation stencil covers
  it — six to eight corners per face, each solve an MP transverse quadrature over
  up to four nodes.
- MHD: MP5/MP7 characteristic reconstruction projects the union of the two
  interface stencils once per face. The left and right states centre on adjacent
  cells and share all but one stencil cell, and both use the same reference state
  and eigensystem, so the shared cells' characteristic amplitudes were being
  derived twice.
- MHD: the auto-`dt` loop no longer reconstructs the live register twice per
  step. `cfl_limit()` already reconstructs both normals to obtain the stable
  timestep, and the first SSP-RK3 residual stage then reconstructed the same
  unchanged register into the same buffers; that result is now reused. The cache
  is keyed on register identity, a mutation generation counter, and the
  reconstruction order, so any state write, ghost refill, background edit,
  escaped mutable reference, or positivity-controller order change drops it.
- HIP: kernels launched at the standard 256-thread block now declare
  `__launch_bounds__(256)`. Without it the compiler must budget registers for the
  1024-thread maximum, which caps a kernel at 128 VGPRs and forces spills to
  scratch to stay under a limit no launch actually needs. This removes the
  register spilling in the PIC gather/push and deposit kernels entirely.
- MHD: the per-step CFL stable-timestep scan and the `div(B)` L-infinity
  diagnostic are now device reductions, removing the per-step copy of the whole
  field back to the host that those host-side scans required.
- MHD: the auto-`dt` CLI run loop no longer computes the CFL stable-step device
  reduction twice per step — a new `MhdSolver2D::step_unchecked` reuses the limit
  the loop just computed instead of re-reducing inside `step`. The `div(B)`
  diagnostic reduction is sampled on the diagnostics cadence (plus the final step)
  rather than every step.
- PIC: per-step full-grid scratch allocations hoisted out of the current filter
  and particle compaction; the Biot–Savart double-precision path uploads the host
  SoA directly instead of an element-by-element copy.
- PIC: the particle gather computes its shape weights once per particle (was four
  times) and gathers self + external E/B in a single stencil sweep; the
  charge-conserving deposit computes each window node index once across the
  Jx/Jy/Jz loops.
- PIC: the per-logged-step alive-particle count reuses the species compaction
  counter instead of allocating a per-block reduction buffer each call.
- PIC: the current deposit no longer synchronizes the device every step. It
  accumulates a persistent overflow flag that the solver drains on a cadence and
  at `finalize()`, removing the per-step host-device round-trip that drained the
  GPU pipeline mid-step (the fatal "reduce dt" overflow is now reported up to one
  cadence interval late).
- Magnetostatics: transient Biot–Savart output buffers skip the allocation
  zero-fill (the kernel overwrites them in full), via a new
  `device_alloc_uninit` / `DeviceBuffer(n, uninitialized)` path.
