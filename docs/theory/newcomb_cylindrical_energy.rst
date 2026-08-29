Cylindrical Newcomb energy functional
======================================

This note fixes the conventions used by Quasar's cylindrical ideal-MHD
stability gate.  They are transcribed from Newcomb's original derivation [1]_,
Eqs. (15)--(18), rather than inferred from a later implementation.  The
conventions matter: changing the radial displacement variable or hiding a
factor of ``r`` in the integration measure changes both coefficient functions.

Perturbation and radial variable
--------------------------------

Newcomb takes the physical displacement components to have real radial
amplitudes

.. math::

   (\xi_r,\; i\xi_\theta,\; i\xi_z)
   \exp\!\left[i(m\theta+kz)\right]

and then defines ``xi`` to mean :math:`\xi_r` itself (his Eq. (6a)).  It is
**not** :math:`r\xi_r`.  After the tangential components are minimized and the
real part of the perturbation is taken, the energy per unit axial length is

.. math::
   :label: newcomb-energy-si

   \frac{\delta W}{L}
   = \frac{\pi}{2}\int_a^b
     \left[f_{\rm SI}(r)\left(\frac{d\xi_r}{dr}\right)^2
          +g_{\rm SI}(r)\xi_r^2\right]dr.

The measure is therefore plain :math:`dr`, not :math:`r\,dr`.  A complex
Hermitian-amplitude convention without taking the real part gives twice this
normalization.  That common positive factor does not affect the sign test, but
the energy and inertia forms must use the same convention in an eigenproblem.
For the fixed conducting-wall problem in the source,
:math:`\xi_r(a)=\xi_r(b)=0` (Newcomb's Eq. (11)).

Quasar's current scalar gate is restricted to an **annulus** with
:math:`a>0`.  A domain that reaches :math:`r=0` is rejected.  The magnetic
axis is not an ordinary conducting endpoint: its admissible behavior depends
on ``m`` and requires a separate mode-dependent regularity derivation.  That
axis contract is intentionally not guessed here.

The Chebyshev coordinate need not itself be ``r``.  If its label is ``s``
(for example normalized flux), the caller supplies the physical radius and
the positive coordinate Jacobian

.. math::

   J_r = \frac{dr}{ds}>0.

The assembled form is then

.. math::

   \int \left[
     \frac{f}{J_r}\left(\frac{d\xi_r}{ds}\right)^2
     +gJ_r\xi_r^2\right]ds.

Thus a normalized-flux quadrature can never be silently treated as ``dr``.
A basis built directly in physical radius uses :math:`J_r=1`.

Coefficient definitions
-----------------------

For compactness define

.. math::

   D = m^2+k^2r^2,\qquad
   C_+ = krB_z+mB_\theta,\qquad
   C_- = krB_z-mB_\theta,

and

.. math::

   F = \frac{mB_\theta}{r}+kB_z,
   \qquad C_+=rF.

With the Fourier sign above, Newcomb's Eqs. (16) and (18), restored to SI
units, are

.. math::
   :label: newcomb-f-si

   f_{\rm SI}(r)
   = \frac{r C_+^2}{\mu_0D}
   = \frac{r^3F^2}{\mu_0D},

.. math::
   :label: newcomb-g-si

   g_{\rm SI}(r)
   = \frac{2k^2r^2}{D}\frac{dp}{dr}
     +\frac{C_+^2}{\mu_0r}\frac{D-1}{D}
     +\frac{2k^2r}{\mu_0D^2}
       \left(k^2r^2B_z^2-m^2B_\theta^2\right).

The rational surface is :math:`F=0`.  In Newcomb's normalization
:math:`\nabla\times\mathbf B=\mathbf J`; equivalently his pressure variable is
:math:`P=\mu_0p_{\rm SI}` and the physical energy is his energy divided by
:math:`\mu_0`.  Equations :eq:`newcomb-f-si` and :eq:`newcomb-g-si` have already
applied that conversion.

Newcomb also gives an equivalent derivative form (his Eq. (17)):

.. math::

   g_{\rm SI}(r)
   = \frac{C_-^2}{\mu_0rD}
     +\frac{C_+^2}{\mu_0r}
     -\frac{2B_\theta}{\mu_0r}\frac{d(rB_\theta)}{dr}
     -\frac{1}{\mu_0}\frac{d}{dr}
       \left[\frac{k^2r^2B_z^2-m^2B_\theta^2}{D}\right].

The pressure-gradient and derivative forms are equivalent only when the
cylindrical equilibrium force balance used in the derivation is satisfied.
Quasar should evaluate Eq. :eq:`newcomb-g-si` for the production coefficient:
it consumes the equilibrium's :math:`dp/dr` directly and avoids numerically
differentiating magnetic-field products.  The derivative form is useful as an
independent manufactured-equilibrium check.

At a rational surface, the two adjacent subdomains retain independent
one-sided values of :math:`\xi_r` for the resonant ``m``.  The same geometric
interface remains merged for all nonresonant harmonics.  This is why radial
breakpoints carry the complete list of resonant ``m`` values, including when a
nearby resonance has been snapped onto a pre-existing regular breakpoint to
avoid an ill-conditioned sliver domain.

As a compact algebraic check, setting :math:`m=0` reduces the source-normalized
coefficients to Newcomb's separately printed Eqs. (19)--(20):

.. math::

   f=rB_z^2,\qquad
   g=\frac{B_z^2}{r}+2P'+k^2rB_z^2.

The corresponding SI coefficients put :math:`1/\mu_0` on every magnetic term
and replace :math:`P'` by :math:`\mu_0p'` before applying the overall
:math:`1/\mu_0` conversion.  This check fixes the derivative sign, powers of
``r``, and the bare-``dr`` measure independently.

Toroidal sign convention
------------------------

The source uses :math:`\exp[i(m\theta+kz)]`.  In the large-aspect-ratio mapping
of a toroidal perturbation :math:`\exp[i(m\theta-n\phi)]`, take
:math:`z=R_0\phi` and :math:`k=-n/R_0`.  Substituting the opposite sign for
``k`` without also changing the Fourier convention moves the rational surface
and invalidates the cylindrical gate.

Suydam necessary condition
--------------------------

Newcomb's Eq. (5) states the local necessary condition for stability.  In SI
units it is

.. math::
   :label: suydam-si

   \frac{rB_z^2}{8\mu_0}
   \left[\frac{d}{dr}
     \ln\left|\frac{B_\theta}{rB_z}\right|\right]^2
   +\frac{dp}{dr} > 0.

For the large-aspect-ratio safety factor
:math:`q=rB_z/(R_0B_\theta)`, this is equivalently

.. math::

   \frac{rB_z^2}{8\mu_0}
   \left(\frac{q'}{q}\right)^2 + p' > 0.

This criterion is necessary, not sufficient.  The cylindrical test must use a
localized trial displacement to show that violating :eq:`suydam-si` drives the
assembled minimum energy negative; satisfying it alone must not be labelled a
proof of global stability.

The connection to the assembled coefficients can be checked without a global
eigenvalue calculation.  Let :math:`r_s` be a helical rational surface,
:math:`F(r_s)=0`, and write

.. math::

   f(r)=\alpha(r-r_s)^2+O((r-r_s)^3),\qquad
   \alpha=\frac{r_s^3[F'(r_s)]^2}{\mu_0D_s}.

At the resonance,
:math:`F'=kB_z(q'/q)` and the last two terms of
:eq:`newcomb-g-si` vanish.  Consequently

.. math::
   :label: suydam-coefficient-identity

   \frac{\alpha}{4}+g(r_s)
   =\frac{2k^2r_s^2}{D_s}
     \left[
       \frac{r_sB_z^2}{8\mu_0}\left(\frac{q'}{q}\right)^2+p'
     \right]_{r_s}.

The prefactor is positive for a nondegenerate helical resonance, so the local
coefficient sign is exactly the Suydam sign.  The ``1/4`` is the sharp local
Hardy bound for the singular form
:math:`\int[\alpha x^2(\xi')^2+g_s\xi^2]dx`.  A negative right-hand side admits
increasingly localized negative-energy trials on either side of the rational
cut.  A positive right-hand side only passes this local necessary gate; it is
not a sufficient global stability result.

Implementation invariants
-------------------------

* The assembled weak form uses :math:`\xi_r`; a non-radius basis supplies
  :math:`dr/ds` explicitly.
* The magnetic terms carry :math:`1/\mu_0`; the pressure-gradient term does
  not.
* ``F`` follows the declared Fourier convention exactly.
* Rational surfaces are zeros of ``F`` and are one-sided subdomain boundaries
  for their tagged harmonic, not ordinary merged collocation points.
* The current scalar gate requires :math:`r>0`; axis regularity is deferred to
  a mode-aware formulation.
* Any future change of variable, such as :math:`u=r\xi_r`, must be derived as a
  separate transformed functional.  Reusing these ``f`` and ``g`` coefficients
  with the new variable is incorrect.

References
----------

.. [1] W. A. Newcomb, *Hydromagnetic Stability of a Diffuse Linear Pinch*,
   UCRL-5447 (1959), printed pp. 9, 11, and 13--14, Eqs. (5), (6a), (11), and
   (15)--(20),
   https://digital.library.unt.edu/ark:/67531/metadc783756/ .  Journal version:
   *Annals of Physics* **10**, 232--267 (1960),
   https://doi.org/10.1016/0003-4916(60)90023-3 .
