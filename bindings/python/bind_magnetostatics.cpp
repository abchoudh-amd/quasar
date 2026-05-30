// Pybind11 bindings for the Quasar magnetostatics module.
//
// Exposes the core Vec3 + magnetostatics public surface (Filament,
// ConductorSystem, PointCloud, ObservationGrid, PlaneSlice, LineProbe,
// BiotSavartConfig, BiotSavartEvaluator) and the geometry generators.
// `evaluate_B` returns a NumPy (N, 3) array of magnetic flux density.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "quasar/core/types.hpp"
#include "quasar/physics/analytic_fields/dipole.hpp"
#include "quasar/physics/analytic_fields/gradient.hpp"
#include "quasar/physics/analytic_fields/uniform.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/field_evaluator.hpp"
#include "quasar/physics/magnetostatics/geometry.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace py = pybind11;

void bind_pic(py::module_& m);

using ::quasar::Real;
using ::quasar::Vec3;
using ::quasar::magnetostatics::BiotSavartConfig;
using ::quasar::magnetostatics::BiotSavartEvaluator;
using ::quasar::magnetostatics::circular_loop;
using ::quasar::magnetostatics::ConductorSystem;
using ::quasar::magnetostatics::Filament;
using ::quasar::magnetostatics::generic_polyline;
using ::quasar::magnetostatics::helix;
using ::quasar::magnetostatics::LineProbe;
using ::quasar::magnetostatics::ObservationGrid;
using ::quasar::magnetostatics::PlaneSlice;
using ::quasar::magnetostatics::PointCloud;
using ::quasar::magnetostatics::polygon;
using ::quasar::magnetostatics::racetrack;
using ::quasar::magnetostatics::solenoid;

namespace {

// Pack a Field<Vec3> result into a NumPy (N, 3) array of Real.
py::array_t<Real> field_to_numpy(const ::quasar::Field<Vec3>& field) {
  const py::ssize_t n = static_cast<py::ssize_t>(field.size());
  py::array_t<Real> arr({n, py::ssize_t{3}});
  auto view = arr.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < n; ++i) {
    view(i, 0) = field[static_cast<std::size_t>(i)].x;
    view(i, 1) = field[static_cast<std::size_t>(i)].y;
    view(i, 2) = field[static_cast<std::size_t>(i)].z;
  }
  return arr;
}

// Pack a Field<Mat3x3> Jacobian into a NumPy (N, 3, 3) array. The (i, j)
// entry is dB_i/dp_j at observation point i.
py::array_t<Real> grad_field_to_numpy(
    const ::quasar::Field<::quasar::Mat3x3>& field) {
  const py::ssize_t n = static_cast<py::ssize_t>(field.size());
  py::array_t<Real> arr({n, py::ssize_t{3}, py::ssize_t{3}});
  auto view = arr.mutable_unchecked<3>();
  for (py::ssize_t i = 0; i < n; ++i) {
    const auto& g = field[static_cast<std::size_t>(i)];
    view(i, 0, 0) = g.r0.x;  view(i, 0, 1) = g.r0.y;  view(i, 0, 2) = g.r0.z;
    view(i, 1, 0) = g.r1.x;  view(i, 1, 1) = g.r1.y;  view(i, 1, 2) = g.r1.z;
    view(i, 2, 0) = g.r2.x;  view(i, 2, 1) = g.r2.y;  view(i, 2, 2) = g.r2.z;
  }
  return arr;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Quasar core bindings (magnetostatics submodule attached).";
  m.attr("__version__") = "0.1.0";

  // -- core types ----------------------------------------------------------

  py::class_<Vec3>(m, "Vec3")
      .def(py::init<>())
      .def(py::init<Real, Real, Real>(),
           py::arg("x"), py::arg("y"), py::arg("z"))
      .def_readwrite("x", &Vec3::x)
      .def_readwrite("y", &Vec3::y)
      .def_readwrite("z", &Vec3::z)
      .def("__repr__", [](const Vec3& v) {
        return "Vec3(" + std::to_string(v.x) + ", "
                       + std::to_string(v.y) + ", "
                       + std::to_string(v.z) + ")";
      })
      .def("__eq__", [](const Vec3& a, const Vec3& b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
      });

  m.attr("mu0")          = ::quasar::mu0;
  m.attr("mu0_over_4pi") = ::quasar::mu0_over_4pi;
  m.attr("pi")           = ::quasar::pi;

  // -- magnetostatics submodule -------------------------------------------

  py::module_ ms = m.def_submodule("magnetostatics",
      "Biot-Savart magnetic-field evaluator on HIP.");

  py::class_<Filament>(ms, "Filament")
      .def(py::init<>())
      .def(py::init([](std::string name, Real current_A,
                        std::vector<Vec3> points) {
        Filament f;
        f.name      = std::move(name);
        f.current_A = current_A;
        f.points    = std::move(points);
        return f;
      }),
           py::arg("name") = std::string{}, py::arg("current_A") = Real{0},
           py::arg("points") = std::vector<Vec3>{})
      .def_readwrite("name",      &Filament::name)
      .def_readwrite("current_A", &Filament::current_A)
      .def_readwrite("points",    &Filament::points);

  py::class_<ConductorSystem>(ms, "ConductorSystem")
      .def(py::init<>())
      .def("add",   &ConductorSystem::add, py::arg("filament"))
      .def("size",  &ConductorSystem::size)
      .def("empty", &ConductorSystem::empty)
      .def("__len__", &ConductorSystem::size);

  py::class_<PointCloud>(ms, "PointCloud")
      .def(py::init<>())
      .def("add",
           static_cast<void (PointCloud::*)(Vec3)>(&PointCloud::add),
           py::arg("point"))
      .def("add_many",
           [](PointCloud& self, const std::vector<Vec3>& pts) {
             self.add(std::span<const Vec3>{pts.data(), pts.size()});
           },
           py::arg("points"))
      .def("points", &PointCloud::points,
           py::return_value_policy::reference_internal)
      .def("size",   &PointCloud::size)
      .def("empty",  &PointCloud::empty)
      .def("__len__", &PointCloud::size);

  py::class_<ObservationGrid>(ms, "ObservationGrid")
      .def(py::init<>())
      .def_readwrite("origin",  &ObservationGrid::origin)
      .def_readwrite("spacing", &ObservationGrid::spacing)
      .def_readwrite("dims",    &ObservationGrid::dims)
      .def("size",            &ObservationGrid::size)
      .def("__len__",         &ObservationGrid::size)
      .def("point_at",        &ObservationGrid::point_at,
           py::arg("i"), py::arg("j"), py::arg("k"))
      .def("to_point_cloud",  &ObservationGrid::to_point_cloud);

  py::class_<PlaneSlice>(ms, "PlaneSlice")
      .def(py::init<>())
      .def_readwrite("origin", &PlaneSlice::origin)
      .def_readwrite("u_step", &PlaneSlice::u_step)
      .def_readwrite("v_step", &PlaneSlice::v_step)
      .def_readwrite("nu",     &PlaneSlice::nu)
      .def_readwrite("nv",     &PlaneSlice::nv)
      .def("size",           &PlaneSlice::size)
      .def("__len__",        &PlaneSlice::size)
      .def("point_at",       &PlaneSlice::point_at,
           py::arg("i"), py::arg("j"))
      .def("to_point_cloud", &PlaneSlice::to_point_cloud);

  py::class_<LineProbe>(ms, "LineProbe")
      .def(py::init<>())
      .def_readwrite("start",    &LineProbe::start)
      .def_readwrite("end",      &LineProbe::end)
      .def_readwrite("n_points", &LineProbe::n_points)
      .def("size",           &LineProbe::size)
      .def("__len__",        &LineProbe::size)
      .def("point_at",       &LineProbe::point_at, py::arg("i"))
      .def("to_point_cloud", &LineProbe::to_point_cloud);

  // Kernel tiling is compile-time (per-gfx, via cmake/QuasarLaunchParams.cmake);
  // BiotSavartConfig carries only the device stream, which Python does not set.
  py::class_<BiotSavartConfig>(ms, "BiotSavartConfig")
      .def(py::init<>());

  // Abstract field-evaluator base so concrete evaluators (Biot-Savart + the
  // analytic fields) can be passed polymorphically to the PIC external-field
  // sampler and selected by registry name.
  using ::quasar::numerics::IFieldEvaluator;
  py::class_<IFieldEvaluator>(ms, "IFieldEvaluator")
      .def("evaluate_B",
           [](const IFieldEvaluator& self,
              const ConductorSystem& cs, const PointCloud& obs) {
             return field_to_numpy(self.evaluate_B(cs, obs));
           },
           py::arg("conductors"), py::arg("observations"))
      .def("evaluate_grad_B",
           [](const IFieldEvaluator& self,
              const ConductorSystem& cs, const PointCloud& obs) {
             return grad_field_to_numpy(self.evaluate_grad_B(cs, obs));
           },
           py::arg("conductors"), py::arg("observations"));

  py::class_<BiotSavartEvaluator, IFieldEvaluator>(ms, "BiotSavartEvaluator")
      .def(py::init<>())
      .def(py::init<BiotSavartConfig>(), py::arg("config"));

  py::class_<::quasar::analytic_fields::UniformEvaluator, IFieldEvaluator>(
      ms, "UniformEvaluator")
      .def(py::init<>())
      .def(py::init<Vec3, Vec3>(), py::arg("b0"), py::arg("e0") = Vec3{0, 0, 0});

  py::class_<::quasar::analytic_fields::DipoleEvaluator, IFieldEvaluator>(
      ms, "DipoleEvaluator")
      .def(py::init<>())
      .def(py::init<Vec3, Vec3>(), py::arg("moment"),
           py::arg("origin") = Vec3{0, 0, 0});

  py::class_<::quasar::analytic_fields::GradientEvaluator, IFieldEvaluator>(
      ms, "GradientEvaluator")
      .def(py::init<>());

  // Create a registered evaluator by name (default-constructed). Mirrors the
  // string-keyed selection the C++ deck boundary uses; concrete evaluators
  // self-register via QUASAR_REGISTER_FIELD_EVALUATOR.
  ms.def("create_field_evaluator",
         [](const std::string& name) {
           return ::quasar::Registry<IFieldEvaluator>::instance().create(name);
         },
         py::arg("name"),
         "Construct a registered IFieldEvaluator by name (e.g. 'biot_savart', "
         "'uniform', 'dipole', 'gradient', 'file_grid').");

  // -- geometry generators -------------------------------------------------

  ms.def("circular_loop", &circular_loop,
         py::arg("center"), py::arg("axis"), py::arg("radius_m"),
         py::arg("n_segments"), py::arg("current_A"),
         py::arg("name") = std::string{"loop"});
  ms.def("helix", &helix,
         py::arg("center"), py::arg("axis"), py::arg("radius_m"),
         py::arg("pitch_m"), py::arg("n_turns"),
         py::arg("n_segments_per_turn"), py::arg("current_A"),
         py::arg("name") = std::string{"helix"});
  ms.def("solenoid", &solenoid,
         py::arg("center"), py::arg("axis"), py::arg("radius_m"),
         py::arg("length_m"), py::arg("n_turns"),
         py::arg("n_segments_per_turn"), py::arg("current_A"),
         py::arg("name") = std::string{"solenoid"});
  ms.def("racetrack", &racetrack,
         py::arg("center"), py::arg("axis"),
         py::arg("straight_length_m"), py::arg("arc_radius_m"),
         py::arg("n_arc_segments"), py::arg("current_A"),
         py::arg("name") = std::string{"racetrack"});
  ms.def("polygon", &polygon,
         py::arg("center"), py::arg("axis"), py::arg("circumradius_m"),
         py::arg("n_sides"), py::arg("current_A"),
         py::arg("name") = std::string{"polygon"});
  ms.def("generic_polyline", &generic_polyline,
         py::arg("points"), py::arg("current_A"),
         py::arg("name") = std::string{"polyline"});

  // pic submodule binds last so it can accept ConductorSystem /
  // BiotSavartEvaluator instances bound above.
  bind_pic(m);
}
