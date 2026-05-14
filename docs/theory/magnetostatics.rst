Magnetostatics: Biot-Savart on thin-wire conductors
====================================================

This page derives the closed-form expressions used in Quasar's
magnetostatics module, both for the magnetic flux density
:math:`\mathbf{B}` and its Jacobian :math:`\nabla \mathbf{B}`. It also
records the discretization-error analysis that justifies the polygon-loop
approximation and the chosen test tolerances.

Continuous Biot-Savart law
--------------------------

For a system of :math:`N_\text{cond}` filamentary conductors, the
magnetic flux density at observation point :math:`\mathbf{r}` is

.. math::

   \mathbf{B}(\mathbf{r}) \;=\;
   \frac{\mu_0}{4\pi}\,
   \sum_{c=1}^{N_\text{cond}}
   I_c \int_{c}
     \frac{d\boldsymbol{\ell}'(\mathbf{r}'_c) \times
           (\mathbf{r} - \mathbf{r}'_c)}
          {\lvert\mathbf{r} - \mathbf{r}'_c\rvert^{3}}.

Quasar approximates each curved conductor as a polyline. The remainder
of this page derives a closed-form expression for the integral over a
single straight segment, valid in :math:`\mathcal{O}(1)` floating-point
operations and free from singularities away from the segment itself.

Closed form for one straight segment
------------------------------------

Let a straight segment run from :math:`\mathbf{a}` to :math:`\mathbf{b}`
carrying current :math:`I`. Parameterize :math:`\mathbf{r}'(t) =
\mathbf{a} + t\,\mathbf{L}` with :math:`\mathbf{L} \equiv \mathbf{b} -
\mathbf{a}` and :math:`t \in [0, 1]`. Define

.. math::

   \mathbf{r}_a = \mathbf{r} - \mathbf{a}, \quad
   \mathbf{r}_b = \mathbf{r} - \mathbf{b}, \quad
   R_a = \lvert\mathbf{r}_a\rvert, \quad
   R_b = \lvert\mathbf{r}_b\rvert.

The Biot-Savart contribution from this segment is

.. math::

   \mathbf{B}_{\text{seg}}(\mathbf{r})
   \;=\;
   \frac{\mu_0 I}{4\pi}\,
   (\mathbf{L} \times \mathbf{r}_a)
   \int_0^1 \frac{dt}{\lvert\mathbf{r}_a - t\,\mathbf{L}\rvert^{3}},

since :math:`\mathbf{L}\times\mathbf{L}=0` removes the dependence of
the cross product on :math:`t`. The remaining integral has a clean
closed form. After expanding
:math:`\lvert\mathbf{r}_a - t\mathbf{L}\rvert^{2}` and applying the
identity

.. math::

   \lvert\mathbf{L}\times\mathbf{r}_a\rvert^{2}
   = (R_a R_b)^{2} - (\mathbf{r}_a\cdot\mathbf{r}_b)^{2}
   = (R_a R_b + \mathbf{r}_a\cdot\mathbf{r}_b)
     (R_a R_b - \mathbf{r}_a\cdot\mathbf{r}_b),

one obtains the **Hanson-Hirshman compact form**

.. math::
   :label: hh

   \mathbf{B}_{\text{seg}}(\mathbf{r})
   \;=\;
   \frac{\mu_0 I}{4\pi}\,
   \frac{R_a + R_b}
        {R_a R_b\,(R_a R_b + \mathbf{r}_a \cdot \mathbf{r}_b)}\,
   (\mathbf{L} \times \mathbf{r}_a).

This is the formula implemented (modulo notation) in
``segment_B`` in
``src/physics/magnetostatics/detail/biot_savart_segment.hpp``. The
denominator vanishes only when :math:`\mathbf{r}` lies on the segment
itself (the integrand is genuinely singular there); the implementation
guards against this with the ``kEps`` floor and returns the zero vector.

Analytic Jacobian
-----------------

The Jacobian :math:`(\nabla \mathbf{B})_{ij} = \partial B_i / \partial
p_j` follows by direct differentiation of :eq:`hh`. Writing
:math:`\mathbf{u}(\mathbf{r}) = \mathbf{L}\times\mathbf{r}_a` and

.. math::

   f(\mathbf{r}) = \frac{R_a + R_b}
                        {R_a R_b\,(R_a R_b + \mathbf{r}_a\cdot\mathbf{r}_b)},

so that :math:`\mathbf{B} = (\mu_0 I / 4\pi)\,\mathbf{u}\,f`,

.. math::

   \frac{\partial B_i}{\partial p_j}
   \;=\; \frac{\mu_0 I}{4\pi}\,
   \Big[
     f\;\underbrace{\frac{\partial u_i}{\partial p_j}}_{[\mathbf{L}]_\times{}_{ij}}
     \;+\;
     u_i\;\frac{\partial f}{\partial p_j}
   \Big],

with the cross-product matrix

.. math::

   [\mathbf{L}]_\times
   \;=\;
   \begin{pmatrix}
     0    & -L_z &  L_y \\
     L_z  &  0   & -L_x \\
    -L_y  &  L_x &  0
   \end{pmatrix}.

The scalar gradient :math:`\nabla f` decomposes through

.. math::

   \begin{aligned}
   \nabla(R_a + R_b)            &= \frac{\mathbf{r}_a}{R_a} + \frac{\mathbf{r}_b}{R_b}, \\
   \nabla(R_a R_b)              &= \frac{R_b}{R_a}\,\mathbf{r}_a
                                  + \frac{R_a}{R_b}\,\mathbf{r}_b, \\
   \nabla(\mathbf{r}_a\cdot\mathbf{r}_b) &= \mathbf{r}_a + \mathbf{r}_b,
   \end{aligned}

which combine to give :math:`\nabla f` via the quotient rule. The
implementation lives in ``segment_gradB`` in the same header. Its
correctness is cross-validated against three independent checks
(``tests/unit/physics/magnetostatics/test_segment_gradient.cpp``):

* central-difference of :math:`\mathbf{B}` agrees with the analytic
  Jacobian to :math:`10^{-6}` relative on a mixed test system,
* the Jacobian is exactly zero for a zero-current source, and
* the trace :math:`\mathrm{tr}(\nabla \mathbf{B}) = \nabla\cdot
  \mathbf{B} \equiv 0` (Maxwell's no-magnetic-monopoles identity).

Discretization error of polyline conductors
-------------------------------------------

When a smooth curve is approximated by an :math:`N`-sided polygon, the
on-axis field of the polygon converges to the circular-loop limit at

.. math::

   B_z^{(N)}(z) - B_z^{(\infty)}(z) \;=\; \mathcal{O}\!\left( N^{-2} \right),

provided the polygon vertices lie on the smooth curve (the case for
``circular_loop`` and ``polygon``). The leading
constant depends on the geometry and the observation point; for the
on-axis circular-loop case the error coefficient is roughly
:math:`(\pi / N)^{2}/6`.

The convergence test
``tests/unit/physics/magnetostatics/test_circular_loop_on_axis.cpp``
verifies the log-log slope :math:`\le -1.8` over :math:`N \in \{32, 64,
128, 256\}` at three axial positions. The off-axis analogue lives in
``test_polyline_convergence.cpp`` and uses :math:`N = 4096` as the
reference instead of a closed form, since off-axis :math:`B` involves
complete elliptic integrals.

Floating-point precision
------------------------

The kernels are templated on a precision parameter ``T`` (Phase 5) and
instantiated for both ``float`` and ``double``. ``Vec3T<T>``,
``Mat3x3T<T>`` and the variable-template constants ``pi_v<T>``,
``mu0_v<T>``, ``mu0_over_4pi_v<T>``, ``kEps_v<T>`` live in
``include/quasar/core/types.hpp``; ``Vec3``, ``Mat3x3``, ``pi`` etc. are
aliases for the ``<double>`` instantiation so the rest of the codebase
compiles unchanged.

Two sibling host classes expose the two precisions:

* :class:`BiotSavartEvaluator` uses ``double`` throughout and returns
  :class:`Field<Vec3>` / :class:`Field<Mat3x3>`. It implements
  :class:`IFieldEvaluator` and is what the registry serves.
* :class:`BiotSavartEvaluatorF` uses ``float`` throughout and returns
  :class:`Field<Vec3f>` / :class:`Field<Mat3x3f>`. Host-side
  conductors / observations stay in ``double``; the evaluator casts to
  ``float`` on upload and the result type makes the precision difference
  visible to callers.

The analytical-reference tolerances are accordingly:

* :math:`10^{-10}` relative for the finite-segment closed form (fp64).
* :math:`10^{-4}` for the :math:`N = 256` circular-loop polygon-vs-exact
  comparison (fp64).
* :math:`5 \times 10^{-6}` relative for fp32-vs-fp64 agreement of
  :math:`\mathbf{B}` on a mixed-system stress test.
* :math:`5 \times 10^{-5}` relative for fp32-vs-fp64 agreement of
  :math:`\nabla\mathbf{B}` on the same test.

References
----------

* J. D. Hanson and S. P. Hirshman, *Compact expressions for the
  Biot-Savart fields of a filamentary segment*, Physics of Plasmas
  **9** (2002), 4410.
* J. D. Jackson, *Classical Electrodynamics*, 3rd ed., chapter 5.
