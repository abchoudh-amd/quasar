Adding an MHD background-field profile
======================================

The ideal-MHD module supports a *field-split* formulation ``B = B0 + b``: a
fixed, discretely divergence-free background field ``B0`` plus an evolved
perturbation ``b``. ``B0`` may be non-uniform and current-carrying; nonzero
curl is not an error. A background-field profile supplies ``B0``. Profiles implement
``quasar::numerics::IMhdBackgroundProfile``
(``include/quasar/numerics/mhd_background_profile.hpp``) and self-register by
name so the input deck selects one with a string. The built-ins are ``"uniform"``
(a spatially constant vector) and ``"linear_vacuum"`` (a linear harmonic
field), implemented and registered in ``src/numerics/mhd_background_profile.cpp``.

The sampling interface is host-only. Configurable profiles also override the
scalar parameter hook::

   virtual Real sample(int comp, Real x, Real y) const = 0;
   virtual bool set_parameter(std::string_view name, Real value);

``comp`` indexes the staggered component the caller wants sampled:

* ``0`` -> ``b0x``, the x-normal component on x-faces,
* ``1`` -> ``b0y``, the y-normal component on y-faces,
* ``2`` -> ``b0z``, the out-of-plane component at cell centers.

The caller passes the *center* of the staggered element for the requested
component; the profile maps element + component to a field value. The interface
is never called from device code — it is sampled host-side at setup time to fill
the background buffers (see "How the profile reaches the device" below).
``set_parameter`` rejects every key by default; override it for the finite scalar
values accepted by the profile's deck ``params`` mapping.

.. important::

   **Return the element's finite-volume moment, not a point value.** The MHD
   state is a finite-volume discretization: ``b0x``/``b0y`` are stored and
   consumed as *face averages* and ``b0z`` as the equation-native *cell
   average*, exactly like the evolved ``bx_face``/``by_face``/``bz_cell`` they
   are added to. In cylindrical geometry that means the unweighted
   :math:`dr` moment for toroidal ``b0z``/``bz_cell``, not the annular
   :math:`r\,dr` moment used by mass-like variables. For a profile
   that is affine over an element the two coincide, so simply evaluating the
   analytic form at the supplied center is exact — this is why both built-ins
   (``"uniform"``, constant; ``"linear_vacuum"``, linear) are correct as written.

   A profile with nonzero curvature over an element must return the average
   instead: integrate the analytic form over the element, or apply the standard
   midpoint correction :math:`\bar{f} = f(x_c) + \frac{h^2}{24}\nabla^2 f +
   O(h^4)`. Returning a bare midpoint value for a nonlinear profile injects an
   :math:`O(h^2)` projection error into ``B0`` that caps the achieved order at
   two no matter which reconstruction the deck selects. The divergence check
   below will *not* catch this: a midpoint-sampled field can be discretely
   divergence-free and still be the wrong finite-volume moment.

Adding a profile
----------------

Mirror the field-evaluator, pusher, boundary, and current-filter guides:

#. Declare a concrete class deriving from
   ``quasar::numerics::IMhdBackgroundProfile`` and override ``sample``.
#. Register it under a deck-facing name with
   ``QUASAR_REGISTER_MHD_BACKGROUND_PROFILE("<name>", <Class>)`` (the macro is in
   ``include/quasar/core/registry.hpp``).
#. If the profile is configurable, override ``set_parameter`` and return
   ``false`` for every unknown key.
#. Place the definition + registration in a ``.cpp`` under ``src/numerics/``.

There is **no** enum, ``switch``, or pybind11 edit: the deck names the profile
and that name is validated against the live registry. The ``numerics`` module is
declared ``quasar_add_module(numerics REGISTERS ...)`` in
``src/numerics/CMakeLists.txt``; the ``REGISTERS`` flag wraps the static archive
in ``$<LINK_LIBRARY:WHOLE_ARCHIVE,...>`` (see ``cmake/QuasarAddModule.cmake``) so
the namespace-scope static initializer the macro emits is never dropped by the
linker, even though nothing else references the TU.

The divergence-free requirement
--------------------------------

.. warning::

   A background profile **must** be discretely divergence-free on the staggered
   grid. This is a correctness contract, not a convenience: the whole point of
   splitting off ``B0`` is that its fixed discrete-divergence defect is at the
   round-off level before stepping. Then
   ``div_h(B0 + b) = div_h(B0) + div_h(b)`` differs from the CT-controlled
   ``div_h(b)`` only by that accepted, time-independent residual. Curl-free is
   not part of the general background contract; current-carrying backgrounds
   remain supported.

Concretely, on a Cartesian grid the face-staggered ``B0`` must satisfy

.. code-block:: text

   (b0x_face(i+1,j) - b0x_face(i,j)) / dx
 + (b0y_face(i,j+1) - b0y_face(i,j)) / dy  ~= 0

over the interior cells. The ``"uniform"`` profile is trivially divergence-free
(equal opposing faces cancel). Cylindrical grids use the matching annular radial
term :math:`(r_{i+1}B_{r,i+1}-r_iB_{r,i})/\int_{r_i}^{r_{i+1}}r\,dr`.
For a **non-uniform** profile, the recommended
construction is to build the face ``B0`` as the discrete curl of a scalar/vector
potential on the same staggered grid; doing so makes it divergence-free *by
construction* rather than by luck, regardless of resolution.

The precise acceptance criterion is

.. math::

   \frac{\max_{i,j}\left|\widetilde r_{i,j}\right|}
        {\max_{i,j}\left(\left|d_{1,i,j}\right|+
                          \left|d_{2,i,j}\right|\right)}
   \le 1024\,\epsilon_{64}.

In Cartesian geometry :math:`d_1=\Delta_x B_x/\Delta x` and
:math:`d_2=\Delta_y B_y/\Delta y`. Each local face offset is cancelled before
normalization, so even a one-ULP slope on a strong DC field remains visible. In
cylindrical geometry the complete first contribution is
:math:`d_1=\Delta_r B_r/\Delta r+(B_{r,h}+B_{r,l})/(2r_c)`, retaining the
physical :math:`B_r/r` curvature, and :math:`d_2=\Delta_z B_z/\Delta z`.
:math:`\widetilde r` normally equals :math:`d_1+d_2`. It is set to zero as
representational forward error only when both directional terms are nonzero,
have opposite signs, the residual is no more than half
:math:`|d_1|+|d_2|`, and it lies within 1024 times the sum of the
metric-weighted storage ULPs of the four faces. The annular bound uses the
actual :math:`(1+q)` and :math:`(1-q)` face coefficients, so the zero-area axis
face creates no artificial allowance. A one-direction defect or two same-sign
defects can never use this local envelope; in particular, a one-ULP slope on a
large Cartesian DC field is rejected.

A zero stencil reports zero. Using global L-infinity norms prevents harmless
roundoff at a local derivative null from becoming an order-one ratio, while an
isolated slope on an otherwise constant field still has defect one. All
directional, residual, scale, and ULP values remain in scaled
mantissa/exponent form through the decision, making the criterion invariant
under field-unit and uniform coordinate-unit rescaling and safe near the
binary64 exponent limits.
``1024 * epsilon(float64)`` is approximately ``2.274e-13``.

The complete padded field must also match the configured device boundary
closure. Periodic ghosts wrap, wall-normal fields are odd with an exact zero on
the wall face while tangential components are even, and the cylindrical axis
also makes ``B_phi`` odd. This prevents a background that is solenoidal only in
the deep interior from injecting a seam or boundary divergence.

This contract is enforced at both public construction paths. The native solver
samples analytic profiles and validates their padded staggered field. For
``file``, ``a_file``, and inline ``conductors`` input, the Python loader
assembles and checks the buffers at build/seed time
(``background_divergence_linf`` in
``python/quasar/mhd/numerics.py``, called from ``build_background_field`` in
``python/quasar/mhd/io.py``), and the native solver validates them again. A
later ``seed_background`` call invalidates the prior validation, and the
complete three-component field is revalidated immediately before the next CFL,
residual, stepping, or divergence operation consumes it. Thus a direct
C++/binding caller cannot bypass the solenoidal requirement by overwriting
constructor-populated buffers.

The deck block
--------------

A background is selected and configured with a top-level ``background_field:``
block (parsed by ``python/quasar/mhd/io.py``). An absent block leaves the
background disabled, and the solver takes its zero-``B0`` fast path (bit-identical
to the non-split solver).

.. code-block:: yaml

   background_field:
     enabled: true
     profile: uniform        # registry name; defaults to "uniform"
     bx0: 0.0                # uniform-profile constants (b0x, b0y, b0z)
     by0: 0.0
     bz0: 1.0
     # params: { ... }       # finite scalar parameters for a named profile
     # file: b0.npz          # alternatively, load B0 from an npz (resolved
                             # relative to the deck directory)
     # a_file: coil.npz      # alternatively, padded-corner A_xyz_grid
     # conductors: [...]      # alternatively, inline SI coil geometry
     # params: {b_scale: 1.0, vacuum_project: true}

The ``profile`` name is validated against the live registry
(``_core.mhd.registered_mhd_background_profiles()``), so a typo or an
unregistered profile fails fast at deck validation. Use ``bx0/by0/bz0`` for the
``uniform`` profile, a ``params`` mapping for a named analytic profile, or a
``file:`` npz to load a precomputed ``B0`` directly. Inline ``conductors`` use
the shared coil/PIC geometry schema, require ``units: SI``, and are mutually
exclusive with ``file`` and ``a_file``.

How the profile reaches the device
----------------------------------

The background ``B0`` is assembled **host-side**. A native ``MhdConfig`` resolves
the registry profile, applies ``background.params`` (followed by the legacy
uniform ``bx0/by0/bz0`` values), samples the padded staggered mesh, applies the
constant ``profile_scale``, and copies the result into the solver. A frontend may
replace those values through
``seed_background(component, buf)`` for file/vector-potential input. Device
kernels only consume the resulting buffers, so no ``.hip`` translation unit
depends on the profile class and a new profile is a pure host/numerics addition.

.. note::

   ``IMhdBackgroundProfile`` is a different interface from
   ``numerics::IFieldEvaluator`` and is still sampled on the host, as described
   above. Only the evaluator axis moved to device SoA buffers (see
   :doc:`adding_a_field_evaluator`). The inline-``conductors`` background path
   is the one place the two meet: it obtains ``A`` from the Biot-Savart
   evaluator, which now computes entirely on the device and is downloaded once
   at the Python binding boundary before the discrete curl. Moving that
   assembly, and the profile sampling itself, onto the device is separate work.

.. note::

   The Python CLI leaves analytic ``uniform`` and ``linear_vacuum`` profiles in
   the native path above; ``profile_scale`` performs their SI conversion without
   destroying registry capability metadata. The standalone
   ``build_background_field`` helper can still sample any registered profile
   through ``_core.mhd.sample_mhd_background_profile`` for validation and tests.
   The serial CLI uses that helper to assemble explicit ``file``/``a_file`` or
   inline-conductor buffers before calling ``seed_background``. Distributed MHD
   currently rejects inline conductors because its canonical background
   interchange cannot preserve their solver-derived physical halo.

.. important::

   A non-uniform background may be current-carrying. The device momentum flux
   contains the Maxwell stress of the total field ``B0 + b``. The Riemann solver
   constructs reduced split fluxes directly, with the static ``B0`` Maxwell
   stress restored by the finite-volume residual. It never materializes a total
   energy or flux containing ``|B0|^2/2``. For a static, solenoidal background,
   let :math:`\mathbf F'_E` denote the reduced energy flux returned by the split
   Riemann solve and :math:`\mathbf F_B` its induction flux. The energy kernel
   enforces the finite-volume identity

   .. math::

      \dot E' =
      -D_E\!\left(\mathbf F'_E
        +\left\langle\mathbf B_0\mathbin{\cdot}\mathbf F_B\right\rangle_f\right)
      -\left\langle\mathbf B_0\mathbin{\cdot}
        \dot{\mathbf b}_{\rm CT}\right\rangle_V .

   Here :math:`D_E` is the same Cartesian or annular face-divergence operator
   used by the conservative update, and :math:`\dot{\mathbf b}_{\rm CT}` is the
   finalized constrained-transport magnetic rate. Face and volume quadrature
   are matched to the spatial order. The implementation conditions this
   expression around the cell background and accumulates background
   differences, CT/Godunov rate differences, and covariance terms at a common
   exponent. It therefore does not rely on a continuum product rule that the
   discrete CT and flux-divergence operators need not satisfy, and it avoids
   forming an ``O(B0^2)`` intermediate. Nonzero curl is permitted;
   ``background_curl_linf`` is a diagnostic, not an acceptance gate. A constant
   cylindrical toroidal ``B0_phi`` is current-carrying because
   :math:`(\nabla\times B_0)_z=B_{0\phi}/r`, but it is supported when the
   staggered divergence criterion passes.

   A trusted domain-wide curl-free construction proof enables one additional
   well-balanced path; it is never inferred from sampled tolerances.
   Analytic Cartesian profiles may return ``true`` from
   ``globally_curl_free()``. The cylindrical ``a_file`` or inline-conductor
   vacuum projection sets the equivalent native
   ``MhdBackgroundSpec::curl_free`` assertion after its solve, provided the
   uniform toroidal ``bz0`` is zero. ``MhdBackgroundSpec::profile_scale`` applies
   a uniform unit conversion
   inside native sampling, preserving an analytic profile's registry proof
   without component-wise overrides. Explicit samples carrying an
   assertion are checked against all staggered curl components as defense in
   depth. Each directional difference cancels its local field offset first;
   only cancellation between independent derivatives receives the ``1e-8``
   local relative tolerance. A curl component containing one derivative must
   vanish exactly in the represented samples, and an axis-containing
   cylindrical domain requires ``B0_phi=0`` throughout its physical cells.
   This check catches gross contradictions but is neither a mathematical proof
   nor a force-error bound; correctness rests on the trusted construction that
   supplied the assertion. Only then does the momentum operator omit the
   pure-static
   :math:`B_0` Maxwell stress, whose divergence is identically
   :math:`(\nabla\times B_0)\times B_0=0`; all :math:`B_0`--:math:`b` cross
   stresses remain. This avoids an :math:`O(h^p B_0^2)` numerical self-force in
   low-beta vacuum backgrounds. Never claim curl-free for a current-carrying
   profile merely to remove that force: doing so violates the trusted-proof
   contract even if the defense-in-depth sample check does not expose the lie.

Inline ``conductors`` and an ``a_file`` are the two vector-potential inputs. The
loader constructs their in-plane field as a discrete curl of padded-corner
samples, making its staggered divergence telescope; inline mode derives that
corner grid from the solver halo and evaluates Biot--Savart in-process. For
cylindrical annuli, continuum vacuum data sampled at corners generally has a
nonzero discrete vacuum-operator residual. ``params.vacuum_project: true``
optionally fixes ``A_phi`` on the outer boundary of the padded corner grid and
solves for ``psi = r A_phi`` with the same annular differences used to form
``B0``. This is field preparation for a discrete vacuum background, not a
requirement of the split equations. The entire padded interval must have
``r > 0``, and conjugate-gradient non-convergence is a hard ``ValueError``.
Without the flag, the supplied potential is differenced directly; the resulting
current-carrying background is valid when it passes the divergence check.

Energy, CFL, and CT consequences
--------------------------------

The split is physical, so ``B0`` participates in the dynamics: the **total**
field ``B0 + b`` enters the eigensystem, Riemann fluxes, Maxwell stress, energy
flux, and fast-magnetosonic speed used for the CFL timestep. The stored split
energy contains gas energy, kinetic energy, and ``0.5 |b|^2`` rather than the
physical ``0.5 |B0 + b|^2``; the rate transformation above keeps that variable
consistent. Constrained transport evolves only ``b`` and never updates the fixed
``B0``.
