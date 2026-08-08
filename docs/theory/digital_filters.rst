Digital current filters
=======================

Particle noise is reduced through a configurable current-filter pipeline.
``BinomialFilter`` represents the standard 1-2-1 smoothing family and
``CompensatedBinomialFilter`` provides the corresponding sharpening hook. The
public API intentionally owns the filter sequence separately from the deposit
scheme so filtered and unfiltered runs share the same particle and field
integration code paths.

Both built-in smoothers act on the out-of-plane current ``Jz`` alone. In 2D
``Jz`` does not enter the discrete charge continuity relation, so it may be
convolved freely. The in-plane pair ``Jx``/``Jy`` is deliberately left
unchanged: the Esirkepov deposit guarantees that its forward divergence matches
``d rho/dt`` cellwise, and applying a smoothing stencil to the current without
applying the identical operator to both endpoint charge densities would change
``div(J)`` and violate Gauss's law. Smoothing the in-plane pair therefore
requires a continuity-preserving (longitudinal/transverse) projection, not the
separable stencil used here.
