Toroidal ideal-MHD energy and inertia
=====================================

This note fixes the continuum convention for Quasar's toroidal stability
operator.  The plasma term is the full three-component, compressible ideal-MHD
energy; the scalar cylindrical Newcomb functional is only a limiting sign
oracle and must not be promoted into a toroidal operator by analogy.

Continuum weak form
-------------------

For a displacement :math:`\boldsymbol\xi`, define

.. math::

   \mathbf Q_\xi=\nabla\times(\boldsymbol\xi\times\mathbf B),
   \qquad D_\xi=\nabla\cdot\boldsymbol\xi,

and let :math:`\mathbf j=(\nabla\times\mathbf B)/\mu_0` be the physical
current density.  The Hermitian polarization of Glasser's real quadratic
form [1]_ is

.. math::

   a_p(\boldsymbol\eta,\boldsymbol\xi)=\int_{\Omega_p}\!\left[
   \frac{\mathbf Q_\eta^*\!\cdot\mathbf Q_\xi}{\mu_0}
   +\gamma P D_\eta^*D_\xi
   +\frac12\left{
     (\boldsymbol\eta^*\!\cdot\nabla P)D_\xi
     +D_\eta^*(\boldsymbol\xi\cdot\nabla P)
   \right\}
   +\frac12\mathbf j\cdot\left(
     \boldsymbol\eta^*\!\times\mathbf Q_\xi
     +\boldsymbol\xi\times\mathbf Q_\eta^*
   \right)\right]dV .

Thus :math:`\delta W_p[\boldsymbol\xi]=a_p(\boldsymbol\xi,
\boldsymbol\xi)/2`.  This expression is preferred over mechanically turning
the squares in Glasser's expanded Eq. (13) into absolute squares: that printed
equation is for real fields and contains cross terms and perfect derivatives.

The kinetic-energy form is

.. math::

   b(\boldsymbol\eta,\boldsymbol\xi)
   =\int_{\Omega_p}\rho\,
     \boldsymbol\eta^*\!\cdot\boldsymbol\xi\,dV,

and the generalized problem is :math:`A_nx=\omega^2M_nx`.  Both matrices use
the same complex-amplitude normalization; multiplying both by the same
positive constant leaves every generalized eigenvalue unchanged.  Static
Grad--Shafranov equilibrium does not determine :math:`\rho`; consequently a
positive density profile is mandatory input to an eigensolve.  A hidden
unit-density default would give growth rates with unstated and generally wrong
physical units.  This variational interpretation follows the ideal-MHD energy
principle of Bernstein *et al.* [3]_.

Straight-field-line components
------------------------------

Let :math:`(\lambda,\vartheta,\varphi)` be radian coordinates, with
:math:`\lambda=\psi_N`, and define

.. math::

   \mathcal J^{-1}
   =\nabla\lambda\times\nabla\vartheta\cdot\nabla\varphi,
   \qquad dV=\mathcal J\,d\lambda\,d\vartheta\,d\varphi.

The covariant coordinate basis is

.. math::

   \mathbf e_\lambda=\mathcal J\nabla\vartheta\times\nabla\varphi,
   \quad
   \mathbf e_\vartheta=\mathcal J\nabla\varphi\times\nabla\lambda,
   \quad
   \mathbf e_\varphi=\mathcal J\nabla\lambda\times\nabla\vartheta.

Displacement and perturbed-field components below are contravariant:

.. math::

   \boldsymbol\xi=\xi^\lambda\mathbf e_\lambda
                 +\xi^\vartheta\mathbf e_\vartheta
                 +\xi^\varphi\mathbf e_\varphi.

With

.. math::

   S=\frac{d}{d\lambda}\left(\frac{\chi}{2\pi}\right),

Glasser's Eqs. (5), (6), (9), and (11), converted from unit-period to radian
angles as in the toroidal-coordinate treatment of Grimm *et al.* [2]_, give

.. math::

   \mathbf B=S(\nabla\varphi-q\nabla\vartheta)\times\nabla\lambda,
   \qquad
   \mathbf B\cdot\nabla
   =\frac{S}{\mathcal J}
    (\partial_\vartheta+q\partial_\varphi),

.. math::

   \xi_s=S(q\xi^\vartheta-\xi^\varphi),

.. math::

   Q^\lambda
   =\frac{S}{\mathcal J}
    (\partial_\vartheta+q\partial_\varphi)\xi^\lambda,

.. math::

   Q^\vartheta
   =-\frac1{\mathcal J}\partial_\lambda(S\xi^\lambda)
    +\frac1{\mathcal J}\partial_\varphi\xi_s,

.. math::

   Q^\varphi
   =-\frac1{\mathcal J}\partial_\lambda(qS\xi^\lambda)
    -\frac1{\mathcal J}\partial_\vartheta\xi_s,

and

.. math::

   D_\xi=\frac1{\mathcal J}\left[
   \partial_\lambda(\mathcal J\xi^\lambda)
   +\partial_\vartheta(\mathcal J\xi^\vartheta)
   +\partial_\varphi(\mathcal J\xi^\varphi)\right].

For :math:`P=P(\lambda)`, the pressure term uses
:math:`\boldsymbol\xi\cdot\nabla P=P_\lambda\xi^\lambda`.  Norms contract
contravariant components with the **covariant** metric,
:math:`g_{ij}`.  For geometric toroidal angle,
:math:`g_{\lambda\varphi}=g_{\vartheta\varphi}=0` and
:math:`g_{\varphi\varphi}=R^2`.

For one toroidal mode and the convention

.. math::

   \boldsymbol\xi
   =\sum_m\boldsymbol\xi_m(\lambda)
     e^{i(m\vartheta-n\varphi)},

:math:`\partial_\vartheta\rightarrow im`,
:math:`\partial_\varphi\rightarrow-in`, and field-line bending contains
:math:`m-nq`.  Axisymmetric equilibrium coefficients still couple different
``m`` harmonics through their :math:`\vartheta` dependence.

Quasar normalization and orientation
------------------------------------

Quasar stores

.. math::

   \lambda=\psi_N
   =\frac{\psi_Q-\psi_{\rm axis}}
          {\psi_{\rm boundary}-\psi_{\rm axis}},
   \qquad
   \mathbf B_p=\nabla\psi_Q\times\nabla\varphi.

Glasser and Grimm use
:math:`\mathbf B_p=\nabla\varphi\times\nabla(\chi/2\pi)`.  Therefore

.. math::

   \frac{\chi}{2\pi}=-\psi_Q,
   \qquad
   S=\psi_{\rm axis}-\psi_{\rm boundary}.

The factor ``S`` is dimensional and must be retained; inserting normalized
flux into the component equations with ``S = 1`` is incorrect.

The existing contour rays run counter-clockwise, while their accumulated
angle uses the unsigned poloidal-field magnitude.  For the established
positive-current convention the source-compatible angle is
:math:`\vartheta=-\theta^*_{\rm stored}`.  Consequently

.. math::

   g_{\lambda\lambda}=R_\lambda^2+Z_\lambda^2,

.. math::

   g_{\lambda\vartheta}
   =-(R_\lambda R_{\theta^*}+Z_\lambda Z_{\theta^*}),

.. math::

   g_{\vartheta\vartheta}=R_{\theta^*}^2+Z_{\theta^*}^2,
   \qquad g_{\varphi\varphi}=R^2.

The sign is a checked invariant, not merely a naming convention.  Geometry
construction must verify

.. math::

   q=\frac{B^\varphi}{B^\vartheta},
   \qquad S=\mathcal J B^\vartheta,

at every usable point.  A check based only on :math:`|B_p|` cannot detect an
orientation error.  PEST geometry additionally requires
:math:`\mathcal J/R^2` to be a flux function.

Equilibrium profiles and current
--------------------------------

The polynomial profile functions are evaluated at :math:`\lambda` but denote
physical derivatives with respect to :math:`\psi_Q`.  With the GS current
normalization ``profile_scale``, define

.. math::

   P_\lambda=(\psi_{\rm boundary}-\psi_{\rm axis})
             \,\mathtt{profile\_scale}\,
             \mathtt{dp\_dpsi}(\lambda),

.. math::

   P(\lambda)=-\int_\lambda^1 P_s(s)\,ds,

where :math:`P(1)=0`.  Likewise,

.. math::

   F^2(\lambda)=F_{\rm vacuum}^2
   -2(\psi_{\rm boundary}-\psi_{\rm axis})
     \,\mathtt{profile\_scale}\!
     \int_\lambda^1\mathtt{ff\_prime}(s)\,ds.

These integrals are analytic for ``PolynomialProfile`` and must not be routed
through the legacy 257-point nearest-neighbour table.  Let
:math:`(FF')_\lambda=(\psi_{\rm boundary}-\psi_{\rm axis})
\mathtt{profile\_scale}\,\mathtt{ff\_prime}`.  If :math:`F^2>0`, then
:math:`F_\lambda=(FF')_\lambda/F`.

Glasser's equilibrium curl, written without his ambiguous bold-``J`` notation,
is

.. math::

   j^\lambda=0,
   \qquad
   j^\vartheta=-\frac{F_\lambda}{\mu_0\mathcal J},
   \qquad
   j^\varphi=qj^\vartheta-\frac{P_\lambda}{S}.

Free boundary
-------------

For a fixed perfectly conducting flux boundary,
:math:`\xi^\lambda=0`; tangential components receive natural conditions.  A
free-boundary calculation requires all three terms

.. math::

   \delta W=\delta W_p+\delta W_s+\delta W_v,

.. math::

   \delta W_s=\frac12\int_{\partial\Omega_p}|\xi_n|^2
   \,\mathbf n\cdot\left[\!\left[
   \nabla\left(P+\frac{B^2}{2\mu_0}\right)
   \right]\!\right]dS,

.. math::

   \delta W_v=\frac1{2\mu_0}\int_{\Omega_v}
   |\nabla\times\mathbf A|^2dV.

Here the jump is outside minus inside for an outward plasma normal and
:math:`\xi_n=\xi^\lambda/|\nabla\lambda|`.  The surface term is not hidden in
the plasma volume integral.  The vacuum field satisfies
:math:`\nabla\times\nabla\times\mathbf A=0`, with
:math:`\mathbf n\times\mathbf A=-\xi_n\widehat{\mathbf B}_0` at the plasma and
:math:`\mathbf n\times\mathbf A=0` at a perfectly conducting wall.

Implementation invariants
-------------------------

* Assemble the Hermitian polarization above for complex Fourier amplitudes.
* Retain the full compression term and all three displacement components.
* Use covariant metrics for vector norms; stored inverse metrics are not a
  substitute.
* Use positive volume weights but preserve orientation signs in curl and cross
  products.
* Require a positive density profile before forming inertia.
* Treat ``n = 0`` separately; the cited non-axisymmetric reduction omits an
  additional inductive contribution.
* Specify magnetic-axis regularity by harmonic and component before including
  the axis as a collocation point.
* Free-boundary work additionally requires wall topology and conserved vacuum
  flux degrees of freedom; the plasma volume form alone is incomplete.

What an annular truncation does and does not prove
--------------------------------------------------

Because magnetic-axis regularity is not yet available, the implementation
evaluates the volume integral above over an annulus
:math:`\lambda\in[\lambda_{\rm in},\lambda_{\rm out}]` rather than over the whole
plasma.  The outer surface carries the fixed conducting condition
:math:`\xi^\lambda=0`.  The truncated inner surface carries the *natural*
condition of the weak form, which is what a variational formulation imposes when
nothing is prescribed: the inner edge is free to move.

This is worth stating explicitly because it is easy to read an annular
:math:`\delta W` as a conservative estimate, and it is not one in either
direction:

* The trial space is not a subspace of the full-plasma trial space.  A
  displacement that is nonzero at :math:`\lambda_{\rm in}` does not extend to
  the core at the same energy, so a negative annular :math:`\delta W` does not
  by itself exhibit an unstable full-plasma displacement.
* Neither is it a restriction that can only miss instabilities.  The annulus
  omits the core contribution to :math:`\delta W` entirely, and that
  contribution is not sign-definite.

So the sign of the annular :math:`\delta W` bounds the full-plasma
:math:`\delta W` neither from above nor from below.  A classification from this
model is a statement about the model.  The implementation reports
``RadialBoundaryModel`` on every result for exactly this reason, and the
supported value is named ``outer_fixed_inner_natural`` rather than something
that reads like a physical boundary condition.

Removing the caveat means implementing the harmonic- and component-dependent
regularity conditions at :math:`\lambda=0` and admitting the axis as a
collocation point, not refining the existing grid.

References
----------

.. [1] A. H. Glasser, *The direct criterion of Newcomb for the ideal MHD
   stability of an axisymmetric toroidal plasma*, Phys. Plasmas **23**, 072505
   (2016), Eqs. (2)--(14), https://doi.org/10.1063/1.4958328 .  Open manuscript:
   https://www.osti.gov/servlets/purl/1418989 .
.. [2] R. C. Grimm, R. L. Dewar, and J. Manickam, *Ideal MHD Stability
   Calculations in Axisymmetric Toroidal Coordinate Systems*, PPPL-1885,
   Eqs. (5), (8)--(14), and (29)--(32), https://doi.org/10.2172/5154988 .
.. [3] I. B. Bernstein, E. A. Frieman, M. D. Kruskal, and R. M. Kulsrud,
   *An Energy Principle for Hydromagnetic Stability Problems*, Proc. Roy. Soc.
   A **244**, 17--40 (1958), Eqs. (1.33)--(1.36) and (2.12)--(2.18),
   https://doi.org/10.1098/rspa.1958.0023 .
