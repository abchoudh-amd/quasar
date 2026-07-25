PIC normalization
=================

Quasar PIC stores kernel inputs in plasma-normalized units. A
``Normalization`` object is built from a reference density, charge, and mass:

.. math::

   \omega_p = \sqrt{\frac{n_\mathrm{ref} q_\mathrm{ref}^2}
                         {\epsilon_0 m_\mathrm{ref}}}.

A default-constructed normalization (equivalently,
``Normalization::identity()``) is the explicit dimensionless identity mapping;
``Normalization::plasma(...)`` selects the physical plasma scaling below.

Lengths scale by ``c / omega_p``, times by ``1 / omega_p``, velocities by
``c``, electric field by ``m c omega_p / q``, and magnetic field by
``m omega_p / q``. Temperature supplied in electron-volts scales by the
reference rest energy :math:`m_\mathrm{ref}c^2/e` (also expressed in eV), so
``UnitTag.temperature_eV`` is dimensionless internally. Python decks may be
specified in SI or normalized units;
diagnostics can emit both representations.
