Adding a current filter
=======================

Current filters implement ``quasar::numerics::ICurrentFilter`` and are stored
in a ``FilterPipeline``. Filters operate after current deposit and before the
Ampere update, making them orthogonal to particle shape and field order.

Add a filter by declaring the type in ``include/quasar/numerics/filter.hpp``,
implementing host/HIP launch logic in ``src/numerics/filter.cpp`` and
``src/backend/hip/pic/filter_hip.hip``, and registering it with
``QUASAR_REGISTER_CURRENT_FILTER``.

Distributed execution contract
------------------------------

Every registered filter must also implement ``distributed_stencils()``.  The
method describes the filter as one or more
``DistributedFilterStencil`` stages.  For each stage, the distributed PIC
runtime applies the nearest-neighbour stencil ``(a, b, a)`` first along x and
then along y, exchanges tile halos after each axis, and repeats that pair
``passes`` times.  Set ``neighbor_weight`` to ``a`` and
``center_weight`` to ``b``.  The runtime validates that ``passes`` is positive
and both weights are finite.

For example, one binomial pass reports:

.. code-block:: cpp

   std::vector<DistributedFilterStencil>
   distributed_stencils() const override {
     return {{n_passes_, Real{0.25}, Real{0.5}}};
   }

The distributed stencil path smooths ``Jz`` only, matching the built-in
filters' charge-conserving contract; ``Jx`` and ``Jy`` are not modified.  A
filter that changes the in-plane current, uses a non-separable stencil, or
requires other state must not return an approximation.  Until it provides an
equivalent distributed implementation, make ``distributed_stencils()`` throw a
descriptive exception so construction fails collectively.

Returning an empty vector means that the filter is an identity operation in a
distributed run.  Use it only for a genuine no-op whose serial ``apply()`` is
also an identity.  It is not an opt-out for a filter that has serial behavior,
because that would make serial and distributed simulations disagree silently.
