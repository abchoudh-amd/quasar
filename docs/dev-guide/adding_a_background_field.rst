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

The caller passes the staggered ``(x, y)`` for the requested component; the
profile only maps position + component to a field value. The interface is never
called from device code — it is sampled host-side at setup time to fill the
background buffers (see "How the profile reaches the device" below).
``set_parameter`` rejects every key by default; override it for the finite scalar
values accepted by the profile's deck ``params`` mapping.

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
   not part of this contract.

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

   \max_{i,j}
   \frac{\left|\sum_k t_{k,i,j}\right|}
        {\sum_k\left|t_{k,i,j}\right|}
   \le 1024\,\epsilon_{64}.

Here :math:`t_k` are the signed face-value/spacing terms in the exact Cartesian
or annular stencil. A zero stencil reports zero. Both sums are formed after
scaling every term to a common binary exponent, making the criterion invariant
under field units, mesh scale, and power-of-two rescaling and safe near the
binary64 exponent limits. ``1024 * epsilon(float64)`` is approximately
``2.274e-13``.

The complete padded field must also match the configured device boundary
closure. Periodic ghosts wrap, wall-normal fields are odd with an exact zero on
the wall face while tangential components are even, and the cylindrical axis
also makes ``B_phi`` odd. This prevents a background that is solenoidal only in
the deep interior from injecting a seam or boundary divergence.

This contract is enforced twice at the public boundaries. The Python deck loader
samples the profile, assembles the staggered ``B0`` buffers, and checks their
interior discrete divergence at build/seed time (``background_divergence_linf``
in ``python/quasar/mhd/numerics.py``, called from ``build_background_field`` in
``python/quasar/mhd/io.py``). The native solver independently validates profiles
sampled by its constructor. A later ``seed_background`` call invalidates that
proof, and the complete three-component field is revalidated immediately before
the next CFL, residual, stepping, or divergence operation consumes it. Thus a
direct C++/binding caller cannot bypass the solenoidal requirement by overwriting
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
     # params: {b_scale: 1.0, vacuum_project: true}

The ``profile`` name is validated against the live registry
(``_core.mhd.registered_mhd_background_profiles()``), so a typo or an
unregistered profile fails fast at deck validation. Use ``bx0/by0/bz0`` for the
``uniform`` profile, a ``params`` mapping for a named analytic profile, or a
``file:`` npz to load a precomputed ``B0`` directly.

How the profile reaches the device
----------------------------------

The background ``B0`` is assembled **host-side**. A native ``MhdConfig`` resolves
the registry profile, applies ``background.params`` (followed by the legacy
uniform ``bx0/by0/bz0`` values), samples the padded staggered mesh, and copies the
result into the solver. A frontend may replace those values through
``seed_background(component, buf)`` for file/vector-potential input. Device
kernels only consume the resulting buffers, so no ``.hip`` translation unit
depends on the profile class and a new profile is a pure host/numerics addition.

.. note::

   Python data path: ``build_background_field`` in ``python/quasar/mhd/io.py``
   assembles the staggered ``B0`` buffers and calls the generic
   ``_core.mhd.sample_mhd_background_profile`` binding at x-faces, y-faces, and
   cell centers. The binding constructs the selected profile from the live
   registry, applies every finite scalar ``params`` entry through
   ``set_parameter``, and invokes ``sample`` over the supplied arrays. Both
   ``uniform`` and ``linear_vacuum`` use this path, and a newly registered
   profile does too without a Python dispatch edit.

.. important::

   A non-uniform background may be current-carrying. The device momentum flux
   contains the Maxwell stress of the total field ``B0 + b``. The Riemann solver
   constructs reduced split fluxes directly, with the static ``B0`` Maxwell
   stress restored by the finite-volume residual. It never materializes a total
   energy or flux containing ``|B0|^2/2``. For a static, solenoidal background,
   the energy kernel directly discretizes

   .. math::

      \partial_t E' + \nabla\mathbin{\cdot}
        (\mathbf F_E-\mathbf B_0\mathbin{\cdot}\mathbf F_B)
      =\mathbf v\mathbin{\cdot}
        [ (\nabla\mathbin{\times}\mathbf B_0)
          \mathbin{\times}(\mathbf B_0+\mathbf b) ].

   The kernel reduces ``curl(B0)`` before multiplication, then accumulates the
   directional flux divergence and expanded current-work terms at a common
   exponent. This retains small equilibrium survivors under dominant-background
   cancellation without forming an ``O(B0^2)`` intermediate for curl-free
   fields. Nonzero curl is permitted; ``background_curl_linf`` is a diagnostic,
   not an acceptance gate. A constant cylindrical toroidal ``B0_phi`` is
   current-carrying because
   :math:`(\nabla\times B_0)_z=B_{0\phi}/r`, but it is supported when the
   staggered divergence criterion passes.

An ``a_file`` is one convenient non-uniform input: the loader constructs the
in-plane field as a discrete curl of the padded-corner vector potential, making
its staggered divergence telescope. For cylindrical annuli, continuum vacuum
data sampled at corners generally has a nonzero discrete vacuum-operator
residual. ``params.vacuum_project: true`` optionally fixes ``A_phi`` on the outer
boundary of the padded corner grid and solves for ``psi = r A_phi`` with the
same annular differences used to form ``B0``. This is field preparation for a
discrete vacuum background, not a requirement of the split equations. The
entire padded interval must have ``r > 0``, and conjugate-gradient
non-convergence is a hard ``ValueError``. Without the flag, the supplied
potential is differenced directly; the resulting current-carrying background is
valid when it passes the divergence check.

Energy, CFL, and CT consequences
--------------------------------

The split is physical, so ``B0`` participates in the dynamics: the **total**
field ``B0 + b`` enters the eigensystem, Riemann fluxes, Maxwell stress, energy
flux, and fast-magnetosonic speed used for the CFL timestep. The stored split
energy contains gas energy, kinetic energy, and ``0.5 |b|^2`` rather than the
physical ``0.5 |B0 + b|^2``; the rate transformation above keeps that variable
consistent. Constrained transport evolves only ``b`` and never updates the fixed
``B0``.
