Adding a new conductor-geometry generator
==========================================

The magnetostatics module ships six geometry generators
(``circular_loop``, ``helix``, ``solenoid``, ``racetrack``, ``polygon``,
``generic_polyline``). This page is a recipe for adding a seventh.

The walk-through tracks three layers that have to stay in lock-step:

1. the C++ free function (host-side polyline construction),
2. the pybind11 export,
3. the YAML schema dispatch.

We will use a hypothetical ``arc_segment`` generator (a single
circular arc between two angles, sitting in the plane perpendicular to
some axis) as the running example. The implementation idiom carries
over to any geometry whose parametric form can be written as host C++.

Step 1 - C++ header and implementation
--------------------------------------

Add the declaration to
``include/quasar/physics/magnetostatics/geometry.hpp``:

.. code-block:: c++

   // Open circular arc of `n_segments` chords in the plane perpendicular
   // to `axis`, centered at `center`, from angle phi_start to phi_end.
   Filament arc_segment(Vec3 center, Vec3 axis, Real radius_m,
                        Real phi_start_rad, Real phi_end_rad,
                        int  n_segments, Real current_A,
                        std::string name = "arc");

Implement it in
``src/physics/magnetostatics/geometry_generators.cpp`` using the
``make_basis`` helper that is already present in that translation unit:

.. code-block:: c++

   Filament arc_segment(Vec3 center, Vec3 axis, Real radius_m,
                        Real phi0, Real phi1, int n_segments,
                        Real current_A, std::string name) {
     check_segment_count("arc_segment", n_segments);
     check_radius("arc_segment", radius_m);
     check_finite("arc_segment", "current_A", current_A);
     const Basis b = make_basis(axis, "arc_segment");

     Filament f{std::move(name), current_A, {}};
     f.points.reserve(static_cast<std::size_t>(n_segments) + 1u);
     for (int k = 0; k <= n_segments; ++k) {
       const Real theta = phi0 + (phi1 - phi0)
                                 * static_cast<Real>(k)
                                 / static_cast<Real>(n_segments);
       f.points.push_back(center + radius_m * (std::cos(theta) * b.u
                                               + std::sin(theta) * b.v));
     }
     return f;
   }

There is intentionally no registry of geometry generators on the C++
side - they are plain free functions. Discovery happens at the YAML/
Python boundary instead (Step 3).

Step 2 - Pybind11 binding
-------------------------

Expose the function in
``bindings/python/bind_magnetostatics.cpp`` alongside the existing
``ms.def(...)`` entries:

.. code-block:: c++

   ms.def("arc_segment", &arc_segment,
          py::arg("center"), py::arg("axis"), py::arg("radius_m"),
          py::arg("phi_start_rad"), py::arg("phi_end_rad"),
          py::arg("n_segments"),  py::arg("current_A"),
          py::arg("name") = std::string{"arc"});

Then re-export it in ``python/quasar/coil/__init__.py``'s ``__all__``
list so ``from quasar.coil import arc_segment`` works.

Step 3 - YAML schema dispatch
-----------------------------

Wire the new ``type`` discriminator into ``_build_geometry`` in
``python/quasar/coil/io.py``:

.. code-block:: python

   if gt == "arc_segment":
       return arc_segment(
           center=_vec3(_require(spec, "center_xyz", "arc_segment")),
           axis=_vec3(_require(spec, "axis_xyz", "arc_segment")),
           radius_m=float(_require(spec, "radius_m", "arc_segment")),
           phi_start_rad=float(_require(spec, "phi_start_rad", "arc_segment")),
           phi_end_rad=float(_require(spec, "phi_end_rad", "arc_segment")),
           n_segments=int(_require(spec, "n_segments", "arc_segment")),
           current_A=current_A,
           name=name,
       )

After this, a deck of the form

.. code-block:: yaml

   conductors:
     - name: half_circle
       current_A: 1.0
       geometry:
         type: arc_segment
         center_xyz:    [0, 0, 0]
         axis_xyz:      [0, 0, 1]
         radius_m:      0.1
         phi_start_rad: 0.0
         phi_end_rad:   3.14159265358979
         n_segments:    64

is automatically accepted by ``quasar coil run``.

Step 4 - Tests
--------------

Two tests are usually enough to anchor a new generator. Both go in
``tests/unit/physics/magnetostatics/test_geometry_generators.cpp``:

* **Geometric invariants** - radius, vertex count, plane, opening
  angle. For ``arc_segment`` we would check that every vertex lies on
  the circle of radius ``radius_m`` perpendicular to ``axis``, and that
  the first/last vertices subtend the expected angle from ``center``.
* **Bad-argument rejection** - assert that
  ``EXPECT_THROW(arc_segment(..., n_segments=0, ...),
  std::invalid_argument)`` (and similar for negative radius, zero
  axis, non-finite current).

If the new generator is going to be exposed through YAML, add a
``test_coil_cli`` parser test in ``tests/python/test_coil_cli.py``
that loads a minimal deck and round-trips it.

Step 5 - Documentation
----------------------

The schema table in
``docs/user-guide/coil_design.rst`` enumerates the recognized
``geometry.type`` values; add the new row there. If the generator is
worth a worked example, drop a deck into ``examples/<name>/`` with a
short ``README.md`` and an integration test in
``tests/python/test_examples.py`` modelled on
``SingleLoopExampleTest``.

Checklist
---------

A merge-ready change to add a generator touches all of:

#. ``include/quasar/physics/magnetostatics/geometry.hpp``
#. ``src/physics/magnetostatics/geometry_generators.cpp``
#. ``bindings/python/bind_magnetostatics.cpp``
#. ``python/quasar/coil/__init__.py``
#. ``python/quasar/coil/io.py``
#. ``tests/unit/physics/magnetostatics/test_geometry_generators.cpp``
#. ``docs/user-guide/coil_design.rst``
