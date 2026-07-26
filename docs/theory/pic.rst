Particle-in-cell discretization
===============================

The ``physics/pic`` module is a 2D3V electromagnetic particle-in-cell
vertical slice. Particle positions live in the ``x-y`` plane while velocities
and fields retain all three components. Fields are stored on a Yee mesh and
advanced with explicit FDTD kernels; particles are advanced with the Boris
pusher.

The implementation exposes two compile-time kernel choices:

* ``fdtd_order = 2`` for the standard Yee curl;
* ``fdtd_order = 4`` for the wider fourth-order centered curl.

Particle gather/deposit uses shape-specialized kernels. ``shape = cic`` uses a
bilinear four-cell support, while ``shape = tsc`` uses a quadratic B-spline
nine-cell support. ``Esirkepov2D`` deposits face current whose forward discrete
divergence satisfies the cellwise continuity equation with the matching charge
shape. For the fourth-order curl, a compact correction maps that second-order
identity onto the fourth-order divergence operator.

On a doubly periodic grid the discrete divergence telescopes exactly, just as
the continuum divergence integrates to zero on a torus.  Consequently a
nonzero net particle charge is incompatible with periodic Gauss's law.  Such a
model must either include explicit counter-species or request the solver's
fixed uniform neutralizing background; silently assuming an ion background
would change the physical problem.

The fixed background cancels only the domain-integrated particle charge. It
does not solve a discrete Poisson problem for finite-particle charge noise, and
the solver does not currently project an arbitrary initial electric field onto
the local Gauss constraint. Users must therefore supply a locally consistent
initial electric field when that constraint is required; the compatible current
deposit preserves, but does not remove, an initial residual. First-order Mur
``outflow`` is likewise restricted to vacuum-field runs because its boundary
correction is not compatible with deposited charge/current.

Time is leapfrogged: positions and electric fields live at integer times, while
evolved particle velocities and magnetic fields live at half times. The public
particle upload and every deck initializer are an explicit startup exception:
their velocities are physical :math:`v(t=0)`, not pre-staggered
:math:`v^{-1/2}`. For a first position step of width :math:`\Delta t`, the
seeded magnetic field remains :math:`B^{-1/2}`. Faraday advances it through the
full interval to :math:`B^{1/2}`, the two magnetic half steps are centred at
:math:`t=0`, and Boris advances the uploaded velocity only over
:math:`\Delta t/2` to :math:`v^{1/2}`. The position drift then uses that centred
velocity.

After startup, if the position step changes, both half-step quantities advance
over ``(dt_previous + dt_current)/2`` and the two bracketing magnetic fields are
interpolated with unequal weights to evaluate the Lorentz force at the integer
time. This permits an exact shortened final position step without breaking time
centering. If the very first step is clipped, its magnetic seed, full Faraday
interval, and half-width particle startup interval all use that clipped width.

``total_em_energy`` reports the positive same-snapshot field norm obtained by
squaring every electric and magnetic Yee sample on its own dual control volume.
It is useful as a field-magnitude diagnostic, but it is not the exactly
conserved vacuum leapfrog invariant because the stored electric and magnetic
fields occupy different times. A conservation proof must instead use the
appropriate cross-time magnetic product (or reconstruct the next magnetic half
step), as the cavity and periodic-wave regressions do.
