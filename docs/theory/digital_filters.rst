Digital current filters
=======================

Particle noise is reduced through a configurable current-filter pipeline.
``BinomialFilter`` represents the standard 1-2-1 smoothing family and
``CompensatedBinomialFilter`` provides the corresponding sharpening hook. The
public API intentionally owns the filter sequence separately from the deposit
scheme so filtered and unfiltered runs share the same particle and field
integration code paths.
