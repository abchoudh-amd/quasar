Adding an MHD background-field profile
======================================

The ideal-MHD module supports a *field-split* formulation ``B = B0 + b``: a
fixed, curl-free background field ``B0`` plus an evolved perturbation ``b``. A
background-field profile supplies ``B0``. Profiles implement
``quasar::numerics::IMhdBackgroundProfile``
(``include/quasar/numerics/mhd_background_profile.hpp``) and self-register by
name so the input deck selects one with a string. The built-in profile is
``"uniform"`` (a spatially constant vector), implemented and registered in
``src/numerics/mhd_background_profile.cpp``.

The interface is a single host-only pure virtual::

   virtual Real sample(int comp, Real x, Real y) const = 0;

``comp`` indexes the staggered component the caller wants sampled:

* ``0`` -> ``b0x``, the x-normal component on x-faces,
* ``1`` -> ``b0y``, the y-normal component on y-faces,
* ``2`` -> ``b0z``, the out-of-plane component at cell centers.

The caller passes the staggered ``(x, y)`` for the requested component; the
profile only maps position + component to a field value. The interface is never
called from device code — it is sampled host-side at setup time to fill the
background buffers (see "How the profile reaches the device" below).

Adding a profile
----------------

Mirror the field-evaluator, pusher, boundary, and current-filter guides:

#. Declare a concrete class deriving from
   ``quasar::numerics::IMhdBackgroundProfile`` and override ``sample``.
#. Register it under a deck-facing name with
   ``QUASAR_REGISTER_MHD_BACKGROUND_PROFILE("<name>", <Class>)`` (the macro is in
   ``include/quasar/core/registry.hpp``).
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
   splitting off ``B0`` is that it contributes nothing to ``div(B)``, so that
   ``div(B0 + b) = div(b)`` stays at round-off and constrained transport only has
   to keep ``div(b)`` clean.

Concretely, the face-staggered ``B0`` must satisfy the discrete face divergence

.. code-block:: text

   (b0x_face(i+1,j) - b0x_face(i,j)) / dx
 + (b0y_face(i,j+1) - b0y_face(i,j)) / dy  ~= 0

over the interior cells. The ``"uniform"`` profile is trivially divergence-free
(equal opposing faces cancel). For a **non-uniform** profile, the recommended
construction is to build the face ``B0`` as the discrete curl of a scalar/vector
potential on the same staggered grid; doing so makes it divergence-free *by
construction* rather than by luck, regardless of resolution.

This contract is enforced. The Python deck loader samples the profile, assembles
the staggered ``B0`` buffers, and checks their interior discrete divergence at
build/seed time (``background_divergence_linf`` in
``python/quasar/mhd/numerics.py``, called from
``build_background_field`` in ``python/quasar/mhd/io.py``). A background whose
interior divergence exceeds a round-off tolerance is **rejected** with a
``ValueError`` before the solver ever sees it — the staging layer cannot seed a
``div(B0) != 0`` background past the solver.

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
     # params: { ... }       # free-form mapping passed to a named profile
     # file: b0.npz          # alternatively, load B0 from an npz (resolved
                             # relative to the deck directory)

The ``profile`` name is validated against the live registry
(``_core.mhd.registered_mhd_background_profiles()``), so a typo or an
unregistered profile fails fast at deck validation. Use ``bx0/by0/bz0`` for the
``uniform`` profile, a ``params`` mapping for a named analytic profile, or a
``file:`` npz to load a precomputed ``B0`` directly.

How the profile reaches the device
----------------------------------

The background ``B0`` is assembled **host-side** and seeded into the solver via
``seed_background(component, buf)``; the device kernels only ever consume the
seeded buffers, so no ``.hip`` translation unit depends on the profile class and a
new profile is a pure host/numerics addition.

.. note::

   Current data path: ``build_background_field`` in ``python/quasar/mhd/io.py``
   assembles the staggered ``B0`` buffers, and the registry is used to **validate
   the profile name**. The ``C++`` ``IMhdBackgroundProfile::sample`` interface
   defines the staggered-sampling contract a future spatially-varying profile will
   implement, but only the ``"uniform"`` profile is wired to a host sampler today
   (``_background_from_profile`` fills constant components; any other registered
   profile raises ``NotImplementedError`` until its host sampler is added). When
   you add a spatial profile, implement its host sampler at that call site (using
   the staggered ``xf``/``yf``/``xc`` meshes) so the assembled buffers match the
   ``sample`` contract.

.. warning::

   **Only a spatially-uniform** ``B0`` **is supported today.** The field-split
   residual uses a conservative-flux-only bookkeeping (the HLLD energy
   back-correction and the total-field Maxwell stress), which is exact only for a
   constant ``B0``. A non-uniform background carries a magnetic-pressure gradient
   and tension that would need explicit static source terms in the
   momentum/energy residual to stay conservative; those are not yet implemented,
   so ``build_background_field`` **rejects a non-uniform** ``B0`` (even a
   divergence-free one) with a ``ValueError``. Adding a non-uniform profile
   therefore also means adding those background source terms — divergence-free
   construction alone is necessary but not sufficient for conservation.

Energy, CFL, and CT consequences
--------------------------------

The split is physical, so ``B0`` participates in the dynamics: the **total**
field ``B0 + b`` enters the MHD fluxes, the total magnetic pressure, the energy
flux, and the fast-magnetosonic speed used for the CFL timestep. What does *not*
change is the bookkeeping that must stay perturbation-only — the stored conserved
energy remains the perturbation magnetic energy ``0.5 |b|^2`` (not
``0.5 |B0 + b|^2``), and constrained transport evolves only ``b`` and never
updates the fixed ``B0``.
