MHD background magnetic field
=============================

Ideal-MHD decks can carry a static **background** (or "guide") magnetic
field via an optional ``background_field:`` block. The model is a static
field split

.. math::

   \mathbf{B} = \mathbf{B}_0 + \mathbf{b},

where the background :math:`\mathbf{B}_0` is a fixed field held constant in
time and the solver evolves only the perturbation
:math:`\mathbf{b}`. Constrained transport advances ``b`` alone;
:math:`\mathbf{B}_0` never changes. This is the natural way to set up a
strongly magnetized plasma without paying the cost of resolving the large
mean field in the evolved state.

The evolved energy is likewise a split variable,

.. math::

   E'=\rho e+\frac{|\mathbf m|^2}{2\rho}+\frac{|\mathbf b|^2}{2}.

It contains neither :math:`|\mathbf B_0|^2/2` nor a
:math:`\mathbf B_0\cdot\mathbf b` cross term.

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
     params: {}              # parameters for a registered analytic profile
     # ... OR load B0 from a file instead of an analytic spec:
     # file: b0.npz          # npz with arrays b0x, b0y, b0z, each
                             # (ny+2g, nx+2g) or flat (storage,)
     # ... OR construct a non-uniform field from corner A_phi:
     # a_file: coil.npz      # npz with padded-corner A_xyz_grid
     # For a_file mode, params may contain b_scale and vacuum_project.

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
``params``       Mapping forwarded to the selected analytic profile. In
                 ``a_file`` mode, ``b_scale`` scales the potential and
                 ``vacuum_project`` requests the annular harmonic projection.
``file``         Path to an ``.npz`` holding a precomputed :math:`\mathbf{B}_0`
                 (arrays ``b0x``, ``b0y``, ``b0z``). Use this **instead of**
                 the analytic spec above.
``a_file``       Path to coil output containing ``A_xyz_grid`` on the full
                 padded corner grid. In cylindrical geometry, lab-``Y`` is
                 interpreted as :math:`A_\phi` and differenced with the annular
                 curl. Mutually exclusive with ``file``.
================ ==============================================================

You supply an analytic profile, ``file:``, or ``a_file:``. Relative paths
are resolved against and confined to the deck directory; absolute paths are
honored as-is.

The built-in ``linear_vacuum`` profile accepts ``gradient`` and ``shear`` and
samples

.. math::

   B_{0x}=g x+s y,\qquad B_{0y}=s x-g y,\qquad B_{0z}=0.

It is divergence-free and curl-free analytically. For a file-loaded background,
each of ``b0x``, ``b0y``, ``b0z`` must match
the grid's storage layout: either the 2-D ghost-padded shape
``(ny + 2g, nx + 2g)`` or the equivalent flat ``(storage,)`` array, where
``g`` is the scheme's ghost width.

.. important::

   Uniform and non-uniform backgrounds are supported, including current-carrying
   fields, provided their staggered face representation is discretely
   divergence-free and compatible with the configured periodic, wall, or axis
   closure. The Riemann solver computes the split momentum and energy fluxes
   directly, without ever forming an :math:`O(|\mathbf B_0|^2)` state or flux.
   For a static, divergence-free background the residual discretizes the split
   conservation law directly,

   .. math::

      \partial_t E' + \nabla\mathbin{\cdot}
        (\mathbf F_E-\mathbf B_0\mathbin{\cdot}\mathbf F_B)
      =\mathbf v\mathbin{\cdot}
        [ (\nabla\mathbin{\times}\mathbf B_0)
          \mathbin{\times}(\mathbf B_0+\mathbf b) ].

   The curl of :math:`\mathbf B_0` is reduced before it is multiplied by either
   field. Directional flux terms and the expanded current-work source share a
   common binary exponent. This retains background-gradient/current work and
   finite survivors even when much larger background terms cancel, while a
   curl-free dominant background never creates an
   :math:`O(|\mathbf B_0|^2)` intermediate.

   ``params.vacuum_project: true`` is an explicit cylindrical-annulus operation.
   It fixes :math:`A_\phi` on the outer boundary of the **padded** corner grid and
   solves the discrete vacuum equation for :math:`\psi=rA_\phi` at interior
   corners. The padded interval must satisfy :math:`r>0`. A
   Jacobi-preconditioned conjugate-gradient solve is required to reach a
   field-derivative-scaled vacuum-operator target; failure is a hard setup
   error.
   Without the flag, the sampled vector potential is used directly. The
   projection is optional field preparation, not a requirement of the split
   equations.

Divergence-free requirement
---------------------------

:math:`\mathbf{B}_0` must also be **discretely divergence-free**. A compatible
uniform vector is trivially solenoidal. Every file-loaded or analytically
sampled :math:`\mathbf{B}_0` is checked against the same discrete face-divergence
operator constrained transport uses, and is **rejected with a clear error**
unless every interior cell satisfies the scale-free stencil test

.. math::

   \max_{i,j}
   \frac{\left|\sum_k t_{k,i,j}\right|}
        {\sum_k\left|t_{k,i,j}\right|}
   \le 1024\,\epsilon_{64},

where :math:`t_k` are the four signed Cartesian face/spacing terms (or the
matching annular-radial and axial terms in cylindrical geometry). A zero stencil
has zero defect. Numerator and denominator share a common binary exponent, so
the test is invariant under field-unit, mesh, and power-of-two rescaling and is
safe near the binary64 exponent limits. Here
``1024 * epsilon(float64)`` is approximately ``2.274e-13``.

The padded background must also be a fixed point of its configured boundary
closure: periodic samples wrap, wall-normal fields vanish on the wall face with
odd normal/even tangential parity, and the cylindrical axis additionally
requires odd :math:`B_\phi`. Pairwise parity uses the same relative round-off
tolerance; the wall/axis normal constraint is exact zero.

The native solver enforces the same check for direct API use. If an application
overwrites any component with ``seed_background``, validation is deferred until
all components have been staged and then runs before the next operation that
uses the background. The cell-centred toroidal component ``b0z`` is checked for
finiteness but does not enter either the divergence or its tolerance scale.

This guarantee matters: because the discrete divergence is linear,

.. math::

   \nabla_h\cdot(\mathbf{B}_0 + \mathbf{b})
   = \nabla_h\cdot\mathbf{B}_0 + \nabla_h\cdot\mathbf{b}
   = \nabla_h\cdot\mathbf{b} + r_0,

so when :math:`\mathbf{B}_0` passes validation, the total-field divergence
differs from the CT-controlled perturbation divergence only by a fixed
round-off-level stencil residual. This is a relative cancellation guarantee,
not a dimensional absolute-error bound.

Physics consequences
--------------------

The background field is **not** inert — the total field
:math:`\mathbf{B}_0 + \mathbf{b}` is what enters the dynamics:

* the **Lorentz force** and **magnetic tension** are computed from the
  total field, so a strong :math:`\mathbf{B}_0` stiffens the fluid against
  transverse bending;
* the **magnetic pressure** :math:`\tfrac{1}{2}|\mathbf{B}_0 + \mathbf{b}|^2`
  uses the total field;
* the total-energy residual is transformed consistently to the stored split
  energy for any static divergence-free background; and
* the **fast-magnetosonic speed** that sets the CFL limit is evaluated on
  the total field.

Because the fast speed grows with :math:`|\mathbf{B}_0|`, enabling a nonzero
background field **tightens** the stable timestep: an ``auto`` (CFL-limited)
``dt`` will be smaller than for the same deck with no background field.

Output convention
-----------------

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
:math:`\mathbf{B}_0 = 0` case. The validated background-divergence residual
stays fixed within the setup acceptance threshold, while the CT-controlled
perturbation divergence stays at round-off.

Run it with:

.. code-block:: bash

   PYTHONPATH=build/hip-gfx942-release/python \
     python -m quasar.mhd.cli run examples/mhd_guide_field/input.yaml

See ``examples/mhd_guide_field/README.md`` for the physics rationale and the
reference numbers the integration test checks against.
