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
nine-cell support. The current deposit entry point is named
``Esirkepov2D``; the present implementation provides the HIP specialization and
keeps the interface ready for a stricter charge-conserving variant.
