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

Time is leapfrogged: positions and electric fields live at integer times, while
particle velocities and magnetic fields live at half times. If the position
step changes, the half-step quantities advance over
``(dt_previous + dt_current)/2`` and the two bracketing magnetic fields are
interpolated with unequal weights to evaluate the Lorentz force at the integer
time. This is what permits an exact shortened final position step without
breaking time centering.
