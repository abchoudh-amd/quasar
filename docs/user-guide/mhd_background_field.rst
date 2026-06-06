MHD background magnetic field
=============================

Ideal-MHD decks can carry a static **background** (or "guide") magnetic
field via an optional ``background_field:`` block. The model is a static
field split

.. math::

   \mathbf{B} = \mathbf{B}_0 + \mathbf{b},

where the background :math:`\mathbf{B}_0` is a fixed, curl-free field held
constant in time and the solver evolves only the perturbation
:math:`\mathbf{b}`. Constrained transport advances ``b`` alone;
:math:`\mathbf{B}_0` never changes. This is the natural way to set up a
strongly magnetized plasma without paying the cost of resolving the large
mean field in the evolved state.

When the block is absent, or ``enabled: false``, there is no background
field and the run is identical to before (default behavior).

Deck block
----------

.. code-block:: yaml

   background_field:
     enabled: true
     profile: uniform        # registry name; default "uniform"
     bx0: 0.0                # uniform-vector components (profile == uniform)
     by0: 0.0
     bz0: 1.0
     params: {}              # free dict for non-uniform named profiles
     # ... OR load B0 from a file instead of an analytic spec:
     file: b0.npz            # npz with arrays b0x, b0y, b0z, each
                             # (ny+2g, nx+2g) or flat (storage,)

Keys
~~~~

================ ==============================================================
Key              Meaning
================ ==============================================================
``enabled``      Master switch. Absent block or ``enabled: false`` means no
                 background field (default, identical to a deck without the
                 block).
``profile``      Registry name of the background-field profile. Validated
                 against the live registry; ``uniform`` is built in and is the
                 default. An unknown name is rejected with an error.
``bx0`` ``by0``  Components of the uniform vector :math:`\mathbf{B}_0` when
``bz0``          ``profile == uniform``. Default ``0`` each.
``params``       Free mapping forwarded to non-uniform named profiles. Empty
                 for ``uniform``.
``file``         Path to an ``.npz`` holding a precomputed :math:`\mathbf{B}_0`
                 (arrays ``b0x``, ``b0y``, ``b0z``). Use this **instead of**
                 the analytic spec above.
================ ==============================================================

You supply **either** an analytic spec (a uniform vector) **or** a ``file:``.
A ``file`` path is resolved relative to the deck directory; an absolute path is
honored as-is.

For a file-loaded background, each of ``b0x``, ``b0y``, ``b0z`` must match
the grid's storage layout: either the 2-D ghost-padded shape
``(ny + 2g, nx + 2g)`` or the equivalent flat ``(storage,)`` array, where
``g`` is the scheme's ghost width.

.. important::

   **Only a spatially-uniform** :math:`\mathbf{B}_0` **is supported today.** The
   solver evolves the perturbation with a conservative-flux-only field split that
   is exact only for a constant background; a spatially-varying
   :math:`\mathbf{B}_0` would need extra source terms (a background
   magnetic-pressure gradient and tension) to conserve energy and momentum, and
   those are not yet implemented. A non-uniform background — whether from a named
   profile or a ``file:`` — is **rejected with a clear error**, even if it is
   divergence-free. Use ``bx0/by0/bz0`` (or a file whose arrays are constant per
   component) to set a uniform guide field. ``profile``/``params`` for a
   non-uniform analytic profile are reserved for a future release.

Divergence-free requirement
---------------------------

:math:`\mathbf{B}_0` must also be **discretely divergence-free**. A uniform
vector is trivially solenoidal and is always accepted. A file-loaded
:math:`\mathbf{B}_0` is checked against the same discrete face-divergence
operator constrained transport uses, and is **rejected with a clear error**
if its divergence exceeds round-off.

This guarantee matters: because the discrete divergence is linear,

.. math::

   \nabla\cdot(\mathbf{B}_0 + \mathbf{b}) = \nabla\cdot\mathbf{b},

so as long as :math:`\mathbf{B}_0` is divergence-free to round-off, the
solenoidal constraint on the total field reduces to the one constrained
transport already enforces on the evolved perturbation.

Physics consequences
--------------------

The background field is **not** inert — the total field
:math:`\mathbf{B}_0 + \mathbf{b}` is what enters the dynamics:

* the **Lorentz force** and **magnetic tension** are computed from the
  total field, so a strong :math:`\mathbf{B}_0` stiffens the fluid against
  transverse bending;
* the **magnetic pressure** :math:`\tfrac{1}{2}|\mathbf{B}_0 + \mathbf{b}|^2`
  uses the total field;
* the conserved **total-energy** budget accounts for the total field; and
* the **fast-magnetosonic speed** that sets the CFL limit is evaluated on
  the total field.

Because the fast speed grows with :math:`|\mathbf{B}_0|`, enabling a nonzero
background field **tightens** the stable timestep: an ``auto`` (CFL-limited)
``dt`` will be smaller than for the same deck with no background field.

Output convention
----------------

Constrained transport evolves only the perturbation, so the
``state_b*`` arrays in the output ``.npz`` are the evolved **perturbation**
:math:`\mathbf{b}`, and any stored magnetic energy is the
perturbation-only :math:`\tfrac{1}{2}|\mathbf{b}|^2`. To recover the total
field, add the background back:

.. math::

   \mathbf{B}_\text{total} = \mathbf{B}_0 + \mathbf{b}.

Worked example
--------------

``examples/mhd_guide_field/`` seeds a small-amplitude, circularly-polarized
Alfvén wave traveling along ``x`` in a fully periodic box, on top of a
static uniform guide field :math:`\mathbf{B}_0 = (B_{x0}, 0, 0)` supplied by
``background_field:``. The Alfvén-wave initial condition seeds only the
transverse perturbation (its own in-plane background is set to zero), so the
mean in-plane field comes entirely from the background block. The guide
field raises the fast speed, so the CFL-limited ``dt`` is tighter than the
:math:`\mathbf{B}_0 = 0` case, while ``div(B0 + b) = div(b)`` stays at
round-off.

Run it with:

.. code-block:: bash

   PYTHONPATH=build/hip-gfx942-release/python \
     python -m quasar.mhd.cli run examples/mhd_guide_field/input.yaml

See ``examples/mhd_guide_field/README.md`` for the physics rationale and the
reference numbers the integration test checks against.
