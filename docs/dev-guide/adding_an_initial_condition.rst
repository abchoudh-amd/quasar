Adding an MHD initial condition
===============================

The deck's ``initial.type`` selects one of six benchmark generators. They are
**device kernels**, not host code and not registry objects, and this page is the
source of truth for adding a seventh.

Why there is no registry here
-----------------------------

Every other pluggable axis in the tree self-registers a class through
``core/registry.hpp`` and is selected by string. The initial conditions are
selected by string too -- the deck is unchanged -- but the dispatch is an
enumerator inside one kernel.

The reason is the same one that lowers equilibrium's profiles to a
``ProfileCoefficients`` POD and the MHD background profiles to an affine POD: a
registry entry is a host object with a vtable, and a vtable cannot cross to the
device. The alternatives were one kernel launch per generator (a launch and a
compilation unit each, to evaluate an analytic expression once per cell) or a
host-side sampler (which is exactly what this replaced). A ``switch`` in a
device function costs one branch that every thread in a launch takes the same
way.

What that means for you: adding a generator touches four places, and none of
them is a registration macro.

The four steps
--------------

**1. An enumerator and a name.** Add to ``MhdInitialConditionKind`` in
``include/quasar/physics/mhd/initial_conditions.hpp``, and to ``kNames`` in
``src/physics/mhd/initial_conditions.cpp``, **in the same order**. The two are
matched by index and there is a test that says so
(``MhdInitialConditions.RegisteredNamesRoundTripThroughTheirEnumerators``). The
name is what the deck writes and what
``_core.mhd.registered_initial_conditions()`` reports, so the Python validator
picks it up with no further edit.

**2. Parameters.** Add named fields to ``MhdInitialConditionSpec``, grouped
under a comment naming your generator. This is a flat tagged block rather than a
union: a union would cost the bindings a per-kind setter, and at this size the
grouping comment carries the same information.

Every field must be a deck scalar or an ``O(1)`` reduction of deck scalars.
Nothing per-cell belongs here. If you find yourself wanting to precompute an
array on the host, that is the signal that the work belongs in the kernel.

**3. The profile.** Add a ``__device__`` function returning a ``Primitive`` in
``src/backend/hip/mhd/mhd_initial_conditions.hip`` and a case in ``evaluate()``.
You get the cell-centre coordinates; you return ``rho``, ``p``, the three
velocity components, and the three magnetic components in **deck units**. The
surrounding kernel handles the primitive-to-conserved assembly, the unit
conversion, and the admissibility checks.

**4. Structural validation.** Add a case to ``validate_kind()`` in
``src/physics/mhd/initial_conditions.cpp`` for the preconditions a kernel cannot
usefully report. A bad radius pair makes every cell wrong in the same way, so
naming the parameter on the host is far more diagnosable than a status bit that
can only say "somewhere". Per-cell failures -- a non-positive density or
pressure at some coordinate -- are the kernel's job and already have bits.

Then extend ``_initial_condition_spec`` in ``python/quasar/mhd/io.py`` to lower
your deck parameters, and add a ``_validate_*_params`` for the deck-level
schema check. Add an ``examples/<case>/`` directory with a README and the
matching entry in ``tests/python/test_examples.py``.

What the staggering contract requires
-------------------------------------

The seed writes ``bx``/``by`` into the **face-staggered** slots and ``bz`` into
the cell-centred slot. Two consequences that are easy to get wrong:

*Divergence.* The staggered discrete divergence must vanish. All six built-ins
achieve this the same way -- each in-plane component is constant along its own
staggering axis (``Bx`` depends only on ``y``, ``By`` only on ``x``), so the
face differences cancel identically. If your field varies along its own normal,
you must construct it as the discrete curl of a corner potential instead, or the
solver will reject it.

*Element moments.* The stored value is a finite-volume moment, not a point
value. A profile that is affine over an element (or piecewise constant away from
its jumps) has moment equal to centre value and can be sampled directly. A
smooth nonlinear profile cannot: sampling it at the centre injects an
``O(h^2)`` projection error that caps the scheme's order regardless of the
reconstruction selected. ``orszag_tang`` does exactly this and is documented as
a midpoint projection for that reason; ``alfven_wave`` is the exactly-projected
case, and it carries an explicit ``sinc(k dx/2)`` factor plus a sub-cell energy
correction to be so. Decide which of those two you are, and say so in the
docstring.

The two-pass assembly
---------------------

The seed is built in two passes and your generator only participates in the
first:

1. primitives to conserved, in deck magnetic units, with the raw face slots in
   the energy; the deck-unit magnetic half-norm is kept in scratch and the
   magnetic components are written already scaled to the solver's
   ``B/sqrt(mu0)`` variable;
2. collocate ``bx_face``/``by_face`` to the cell with the solver's own
   ``mhd_staggering.hpp`` quadrature, swap the raw magnetic energy for the
   collocated one, and run the positivity preflight on the exact conserved
   arrays pass 1 produced.

Pass 2 exists so the seeded energy and the solver's EOS read the same ``B`` by
construction. For all six current generators it is a no-op to round-off, for the
same reason their divergence vanishes -- see
``tests/unit/physics/mhd/test_mhd_initial_conditions.cpp``, which pins both that
identity and a case where the pass genuinely changes the answer. Do not delete
it as dead code.

The preflight runs on the conserved arrays rather than on the primitives they
came from, deliberately: the collocation and the float64 energy assembly can
expose a loss of internal energy that a check on the primitives cannot see, and
finding it here beats finding it in the first CFL reduction.

Coordinates
-----------

Use ``Grid2D::x_at_cell_center`` / ``y_at_cell_center``. They are FMAs, and they
are the same mapping the solver uses for every geometric factor, so a seed built
from them is consistent with the mesh rather than merely close to it. Do not
reconstruct ``origin + (i + 0.5) * dx`` by hand.

Ghost cells are seeded too. Every generator is an analytic profile of the
coordinate, so a ghost cell carries the value the interior would, and the halo
is consistent before the first boundary fill rather than after it.
