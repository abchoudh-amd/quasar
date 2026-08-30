// Pybind11 bindings for the Quasar magnetostatics module.
//
// Exposes the core Vec3 + magnetostatics public surface (Filament,
// ConductorSystem, PointCloud, ObservationGrid, PlaneSlice, LineProbe,
// BiotSavartConfig, BiotSavartEvaluator) and the geometry generators.
// `evaluate_B` returns a NumPy (N, 3) array of magnetic flux density.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "quasar/core/device_observations.hpp"
#include "quasar/core/field_source.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/analytic_fields/dipole.hpp"
#include "quasar/physics/analytic_fields/file_grid.hpp"
#include "quasar/physics/analytic_fields/gradient.hpp"
#include "quasar/physics/analytic_fields/uniform.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/field_evaluator.hpp"
#include "quasar/physics/magnetostatics/geometry.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <array>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

void bind_pic(py::module_& m);
void bind_mhd(py::module_& m);

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
using ::quasar::core::DevicePointCloud;
using ::quasar::magnetostatics::FilamentPoints;
using ::quasar::magnetostatics::PointCloud;
using ::quasar::magnetostatics::polygon;
using ::quasar::magnetostatics::racetrack;
using ::quasar::magnetostatics::solenoid;

namespace {

::quasar::Mat3x3 matrix_from_rows(const std::vector<std::vector<Real>>& rows) {
  if (rows.size() != 3 || rows[0].size() != 3 || rows[1].size() != 3 ||
      rows[2].size() != 3) {
    throw std::invalid_argument{"GradientEvaluator: grad must be a 3x3 matrix"};
  }
  return ::quasar::Mat3x3{
      Vec3{rows[0][0], rows[0][1], rows[0][2]},
      Vec3{rows[1][0], rows[1][1], rows[1][2]},
      Vec3{rows[2][0], rows[2][1], rows[2][2]}};
}

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

  // `magnetostatics` is the deliberate shared home for ALL IFieldEvaluator types
  // (Biot-Savart plus the analytic_fields evaluators Uniform/Dipole/Gradient and
  // the create_field_evaluator factory): they share the single IFieldEvaluator
  // registry, and the PIC external-field sampler consumes them through that one
  // surface. Binding them here rather than in a separate `analytic_fields`
  // submodule keeps that shared registry surface in one place; it is an
  // intentional choice, not accidental coupling.
  py::module_ ms = m.def_submodule("magnetostatics",
      "Field-evaluator surface (Biot-Savart + analytic fields) on HIP.");

  // Filament owns device buffers and is therefore move-only. Python holds it
  // by shared_ptr so `cs.add(circular_loop(...))` still reads naturally; `add`
  // moves out of the holder, which leaves the Python object empty rather than
  // silently duplicating device allocations.
  py::class_<Filament>(ms, "Filament")
      .def(py::init<>())
      .def(py::init([](std::string name, Real current_A,
                        const std::vector<Vec3>& points) {
        Filament f;
        f.name      = std::move(name);
        f.current_A = current_A;
        f.points    = FilamentPoints::upload(points);
        return f;
      }),
           py::arg("name") = std::string{}, py::arg("current_A") = Real{0},
           py::arg("points") = std::vector<Vec3>{})
      .def_readwrite("name",      &Filament::name)
      .def_readwrite("current_A", &Filament::current_A)
      // The vertices live on the device; reading them back is an output
      // boundary, so this is a property returning a copy rather than a
      // reference into the object.
      .def_property_readonly(
          "points",
          [](const Filament& self) { return self.points.to_host(); },
          "Vertices as a list of Vec3, downloaded from the device.")
      .def_property_readonly(
          "n_points",
          [](const Filament& self) { return self.points.size(); });

  // Axis-neutral field-source base so the IFieldEvaluator surface can accept any
  // source polymorphically (the evaluator contract is on core::IFieldSource).
  // ConductorSystem derives from it; analytic evaluators ignore the source.
  py::class_<::quasar::core::IFieldSource>(ms, "IFieldSource");

  py::class_<ConductorSystem, ::quasar::core::IFieldSource>(ms, "ConductorSystem")
      .def(py::init<>())
      .def("add",
           [](ConductorSystem& self, Filament& filament) {
             self.add(std::move(filament));
           },
           py::arg("filament"),
           "Take ownership of a filament. The argument is moved from: its "
           "device vertices belong to the system afterwards.")
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

  // Device-resident observation points, the shape the field evaluators consume.
  // Opaque from Python: it exists to be handed straight back to evaluate_*
  // without a host round trip. Construct one from a structured observation set
  // via `to_device_point_cloud()`, or upload an explicit PointCloud.
  py::class_<DevicePointCloud>(ms, "DevicePointCloud")
      .def(py::init<>())
      .def_static("upload",
                  [](const PointCloud& points) {
                    return DevicePointCloud::upload(points);
                  },
                  py::arg("points"),
                  "Upload a host PointCloud to the device.")
      .def("size",    &DevicePointCloud::size)
      .def("empty",   &DevicePointCloud::empty)
      .def("__len__", &DevicePointCloud::size);

  py::class_<ObservationGrid>(ms, "ObservationGrid")
      .def(py::init<>())
      .def_readwrite("origin",  &ObservationGrid::origin)
      .def_readwrite("spacing", &ObservationGrid::spacing)
      .def_readwrite("dims",    &ObservationGrid::dims)
      .def("size",            &ObservationGrid::size)
      .def("__len__",         &ObservationGrid::size)
      .def("point_at",        &ObservationGrid::point_at,
           py::arg("i"), py::arg("j"), py::arg("k"))
      .def("to_point_cloud",  &ObservationGrid::to_point_cloud)
      .def("to_device_point_cloud", &ObservationGrid::to_device_point_cloud,
           "Expand straight into device SoA planes. Agrees with "
           "point_at() bit for bit and skips the host round trip "
           "that to_point_cloud() + evaluate_* would pay.");

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
      .def("to_point_cloud", &PlaneSlice::to_point_cloud)
      .def("to_device_point_cloud", &PlaneSlice::to_device_point_cloud,
           "Expand straight into device SoA planes. Agrees with "
           "point_at() bit for bit and skips the host round trip "
           "that to_point_cloud() + evaluate_* would pay.");

  py::class_<LineProbe>(ms, "LineProbe")
      .def(py::init<>())
      .def_readwrite("start",    &LineProbe::start)
      .def_readwrite("end",      &LineProbe::end)
      .def_readwrite("n_points", &LineProbe::n_points)
      .def("size",           &LineProbe::size)
      .def("__len__",        &LineProbe::size)
      .def("point_at",       &LineProbe::point_at, py::arg("i"))
      .def("to_point_cloud", &LineProbe::to_point_cloud)
      .def("to_device_point_cloud", &LineProbe::to_device_point_cloud,
           "Expand straight into device SoA planes. Agrees with "
           "point_at() bit for bit and skips the host round trip "
           "that to_point_cloud() + evaluate_* would pay.");

  // Kernel tiling is compile-time (per-gfx, via cmake/QuasarLaunchParams.cmake);
  // BiotSavartConfig carries only the device stream, which Python does not set.
  py::class_<BiotSavartConfig>(ms, "BiotSavartConfig")
      .def(py::init<>());

  // Abstract field-evaluator base so concrete evaluators (Biot-Savart + the
  // analytic fields) can be passed polymorphically to the PIC external-field
  // sampler and selected by registry name.
  using ::quasar::numerics::IFieldEvaluator;
  py::class_<IFieldEvaluator>(ms, "IFieldEvaluator")
      .def_property_readonly(
           "provides_vector_potential",
           &IFieldEvaluator::provides_vector_potential)
      .def_property_readonly(
           "provides_grad_B",
           &IFieldEvaluator::provides_grad_B)
      .def("configure",
           [](IFieldEvaluator& self, const ::quasar::numerics::EvaluatorParams& params) {
             self.configure(params);
           },
           py::arg("params"),
           "Apply deck parameters (name -> flat float list; Vec3=3, Mat3x3=9 "
           "row-major) after registry construction. Unknown keys are rejected.")
      // Python is an output boundary by definition: these hand back NumPy
      // arrays. So each wrapper uploads the host PointCloud, evaluates entirely
      // on the device, and downloads once via .to_host(). The C++ callers that
      // chain evaluations (the PIC external-field sampler, the MHD background
      // builder) keep the DevicePointCloud and never make this round trip.
      .def("evaluate_B",
           [](const IFieldEvaluator& self,
              const ::quasar::core::IFieldSource& src, const PointCloud& obs) {
             const auto device_obs = DevicePointCloud::upload(obs);
             return field_to_numpy(self.evaluate_B(src, device_obs).to_host());
           },
           py::arg("source"), py::arg("observations"))
      .def("evaluate_E",
           [](const IFieldEvaluator& self,
              const ::quasar::core::IFieldSource& src, const PointCloud& obs) {
             const auto device_obs = DevicePointCloud::upload(obs);
             return field_to_numpy(self.evaluate_E(src, device_obs).to_host());
           },
           py::arg("source"), py::arg("observations"))
      .def("evaluate_grad_B",
           [](const IFieldEvaluator& self,
              const ::quasar::core::IFieldSource& src, const PointCloud& obs) {
             const auto device_obs = DevicePointCloud::upload(obs);
             return grad_field_to_numpy(
                 self.evaluate_grad_B(src, device_obs).to_host());
           },
           py::arg("source"), py::arg("observations"))
      .def("evaluate_A",
           [](const IFieldEvaluator& self,
              const ::quasar::core::IFieldSource& src, const PointCloud& obs) {
             const auto device_obs = DevicePointCloud::upload(obs);
             return field_to_numpy(self.evaluate_A(src, device_obs).to_host());
           },
           py::arg("source"), py::arg("observations"),
           "Magnetic vector potential A (B = curl A), NumPy (N, 3). Raises if "
           "the evaluator does not model A.")
      // Overloads taking points that are already on the device. A deck that
      // built its observation set with to_device_point_cloud() passes it
      // straight through, so the coordinates are generated on the device,
      // consumed on the device, and only the field comes back.
      .def("evaluate_B",
           [](const IFieldEvaluator& self,
              const ::quasar::core::IFieldSource& src,
              const DevicePointCloud& obs) {
             return field_to_numpy(self.evaluate_B(src, obs).to_host());
           },
           py::arg("source"), py::arg("observations"))
      .def("evaluate_E",
           [](const IFieldEvaluator& self,
              const ::quasar::core::IFieldSource& src,
              const DevicePointCloud& obs) {
             return field_to_numpy(self.evaluate_E(src, obs).to_host());
           },
           py::arg("source"), py::arg("observations"))
      .def("evaluate_grad_B",
           [](const IFieldEvaluator& self,
              const ::quasar::core::IFieldSource& src,
              const DevicePointCloud& obs) {
             return grad_field_to_numpy(
                 self.evaluate_grad_B(src, obs).to_host());
           },
           py::arg("source"), py::arg("observations"))
      .def("evaluate_A",
           [](const IFieldEvaluator& self,
              const ::quasar::core::IFieldSource& src,
              const DevicePointCloud& obs) {
             return field_to_numpy(self.evaluate_A(src, obs).to_host());
           },
           py::arg("source"), py::arg("observations"));

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
      .def(py::init<>())
      .def(py::init([](Vec3 b0, const std::vector<std::vector<Real>>& grad,
                       Vec3 origin) {
             return ::quasar::analytic_fields::GradientEvaluator{
                 b0, matrix_from_rows(grad), origin};
           }),
           py::arg("b0"), py::arg("grad"), py::arg("origin") = Vec3{0, 0, 0});

  py::class_<::quasar::analytic_fields::FileGridEvaluator, IFieldEvaluator>(
      ms, "FileGridEvaluator")
      .def(py::init<>())
      .def(py::init<std::string>(), py::arg("path"))
      .def_property_readonly("path",
          &::quasar::analytic_fields::FileGridEvaluator::path)
      .def_property_readonly("configured",
          &::quasar::analytic_fields::FileGridEvaluator::configured);

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
  ms.def("field_evaluator_names",
         [] {
           auto names = ::quasar::Registry<IFieldEvaluator>::instance().names();
           std::sort(names.begin(), names.end());
           return names;
         },
         "Return every currently registered field-evaluator name in sorted order.");
  ms.def("field_evaluator_provides_vector_potential",
         [](const std::string& name) {
           const auto evaluator =
               ::quasar::Registry<IFieldEvaluator>::instance().create(name);
           return evaluator->provides_vector_potential();
         },
         py::arg("name"),
         "Return whether a registered evaluator implements magnetic vector "
         "potential A.");
  ms.def("field_evaluator_provides_grad_B",
         [](const std::string& name) {
           const auto evaluator =
               ::quasar::Registry<IFieldEvaluator>::instance().create(name);
           return evaluator->provides_grad_B();
         },
         py::arg("name"),
         "Return whether a registered evaluator supplies a trustworthy "
         "magnetic-field Jacobian.");

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
  // mhd submodule (self-contained; binds its own Grid2D at the mhd scope).
  bind_mhd(m);
}
