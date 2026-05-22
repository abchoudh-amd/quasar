PIC normalization
=================

Quasar PIC stores kernel inputs in plasma-normalized units. A
``Normalization`` object is built from a reference density, charge, and mass:

.. math::

   \omega_p = \sqrt{\frac{n_\mathrm{ref} q_\mathrm{ref}^2}
                         {\epsilon_0 m_\mathrm{ref}}}.

Lengths scale by ``c / omega_p``, times by ``1 / omega_p``, velocities by
``c``, electric field by ``m c omega_p / q``, and magnetic field by
``m omega_p / q``. Python decks may be specified in SI or normalized units;
diagnostics can emit both representations.
