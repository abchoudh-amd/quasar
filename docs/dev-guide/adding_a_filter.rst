Adding a current filter
=======================

Current filters implement ``quasar::numerics::ICurrentFilter`` and are stored
in a ``FilterPipeline``. Filters operate after current deposit and before the
Ampere update, making them orthogonal to particle shape and field order.

Add a filter by declaring the type in ``include/quasar/numerics/filter.hpp``,
implementing host/HIP launch logic in ``src/numerics/filter.cpp`` and
``src/backend/hip/pic/filter_hip.hip``, and registering it with
``QUASAR_REGISTER_CURRENT_FILTER``.
