// Pybind11 bindings for the Quasar ideal-MHD module.
//
// Mirrors bindings/python/bind_pic.cpp: attaches a `_core.mhd` submodule that
// exposes the deck-facing config struct (MhdConfig), the solver driver
// (MhdSolver2D) with the seed/step/readback seam the Python CLI drives, and the
// registry-introspection helpers (registered_*) so the Python deck validators
// select schemes against the live C++ registries instead of a hardcoded mirror.
//
// Grid2D is the shared core type quasar::Grid2D, already bound by bind_pic (which
// runs first). pybind11 registers a class by its C++ TYPE globally, not by
// (module, name), so re-binding quasar::Grid2D here would raise "type already
// registered". Instead this TU ALIASES the existing pybind class into the mhd
// submodule (mhd.Grid2D is the same object as pic.Grid2D), so `_core.mhd.Grid2D`
// and `quasar.mhd.Grid2D` resolve without importing quasar.pic.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numpy_utils.hpp"

#include "quasar/boundary/mhd_boundary.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/numerics/ct_scheme.hpp"
#include "quasar/numerics/flux_reconstruction.hpp"
#include "quasar/numerics/mhd_background_profile.hpp"
#include "quasar/numerics/positivity_limiter.hpp"
#include "quasar/numerics/riemann_solver.hpp"
#include "quasar/numerics/ssprk_integrator.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using ::quasar::Real;
using ::quasar::python_detail::numpy_to_real_vector;
using ::quasar::python_detail::require_real_array;

template <typename T>
py::array_t<T> vector_to_numpy(const std::vector<T>& v) {
  py::array_t<T> arr(static_cast<py::ssize_t>(v.size()));
  if (!v.empty()) {
    std::copy(v.begin(), v.end(), arr.mutable_data());
  }
  return arr;
}

}  // namespace

void bind_mhd(py::module_& m) {
  py::module_ mhd =
      m.def_submodule(
          "mhd", "2D ideal-MHD (finite-volume + constrained transport) scaffolding.");

  // -- Grid -----------------------------------------------------------------
  // Alias the already-registered quasar::Grid2D pybind class (bound by bind_pic,
  // which runs before this) into the mhd submodule. py::type::of<T>() returns the
  // existing class object, so `_core.mhd.Grid2D` is the same type as the pic one
  // (ctor signature nx, ny, lx, ly, origin_x=0, origin_y=0, nghost=1) without a
  // duplicate registration.
  mhd.attr("Grid2D") = py::type::of<quasar::Grid2D>();

  // -- Registry introspection -----------------------------------------------
  // Expose the registered scheme/BC names so the Python deck validators select
  // against the live C++ registry instead of a hardcoded mirror. Sorted for
  // stable error messages (mirrors the registered_* style in bind_pic.cpp).
  auto sorted_names = [](std::vector<std::string> v) {
    // The distributed tile builder installs "internal" directly in native
    // configs.  It is deliberately absent from YAML-facing introspection.
    std::erase(v, "internal");
    std::sort(v.begin(), v.end());
    return v;
  };
  mhd.def("registered_riemann_solvers",
          [sorted_names]() {
            return sorted_names(
                quasar::Registry<quasar::numerics::IRiemannSolver>::instance().names());
          },
          "Names of registered MHD Riemann solvers.");
  mhd.def("registered_reconstructions",
          [sorted_names]() {
            return sorted_names(
                quasar::Registry<quasar::numerics::IFluxReconstruction>::instance().names());
          },
          "Names of registered MHD flux-reconstruction schemes.");
  mhd.def("registered_integrators",
          [sorted_names]() {
            return sorted_names(
                quasar::Registry<quasar::numerics::ISsprkIntegrator>::instance().names());
          },
          "Names of registered SSP-RK integrators.");
  mhd.def("registered_ct_schemes",
          [sorted_names]() {
            return sorted_names(
                quasar::Registry<quasar::numerics::ICtScheme>::instance().names());
          },
          "Names of registered constrained-transport schemes.");
  mhd.def("registered_positivity_limiters",
          [sorted_names]() {
            return sorted_names(
                quasar::Registry<quasar::numerics::IPositivityLimiter>::instance().names());
          },
          "Names of registered positivity limiters.");
  mhd.def("registered_mhd_fluid_boundaries",
          [sorted_names]() {
            return sorted_names(
                quasar::Registry<quasar::boundary::IMhdFluidBoundary>::instance().names());
          },
          "Names of registered MHD fluid (rho, m, energy) boundary conditions.");
  mhd.def("registered_mhd_field_boundaries",
          [sorted_names]() {
            return sorted_names(
                quasar::Registry<quasar::boundary::IMhdFieldBoundary>::instance().names());
          },
          "Names of registered MHD magnetic-field boundary conditions.");
  mhd.def("registered_mhd_background_profiles",
          [sorted_names]() {
            return sorted_names(
                quasar::Registry<quasar::numerics::IMhdBackgroundProfile>::instance().names());
          },
          "Names of registered static background-field (B0) profiles.");
  mhd.def(
      "sample_mhd_background_profile",
      [](const std::string& name, int comp, const py::object& x_input,
         const py::object& y_input, const py::dict& params) {
        const auto x = require_real_array(x_input, "background x coordinates");
        const auto y = require_real_array(y_input, "background y coordinates");
        if (comp < 0 || comp > 2) {
          throw std::invalid_argument{"background component must be 0, 1, or 2"};
        }
        if (x.ndim() != y.ndim()) {
          throw std::invalid_argument{"background x/y arrays must have equal rank"};
        }
        for (py::ssize_t d = 0; d < x.ndim(); ++d) {
          if (x.shape(d) != y.shape(d)) {
            throw std::invalid_argument{"background x/y arrays must have equal shape"};
          }
        }
        auto profile =
            quasar::Registry<quasar::numerics::IMhdBackgroundProfile>::instance().create(name);
        for (const auto& item : params) {
          const std::string key = py::cast<std::string>(item.first);
          const Real value = py::cast<Real>(item.second);
          if (!std::isfinite(value)) {
            throw std::invalid_argument{"background parameter " + key + " must be finite"};
          }
          if (!profile->set_parameter(key, value)) {
            throw std::invalid_argument{"unknown parameter " + key
                                        + " for background profile " + name};
          }
        }
        py::array_t<Real> out(x.request().shape);
        const auto n = static_cast<std::size_t>(x.size());
        const Real* xp = x.data();
        const Real* yp = y.data();
        Real* op = out.mutable_data();
        for (std::size_t k = 0; k < n; ++k) {
          if (!std::isfinite(xp[k]) || !std::isfinite(yp[k])) {
            throw std::invalid_argument{
                "background profile coordinates must be finite"};
          }
          op[k] = profile->sample(comp, xp[k], yp[k]);
          if (!std::isfinite(op[k])) {
            throw std::runtime_error{"background profile produced a non-finite sample"};
          }
        }
        return out;
      },
      py::arg("name"), py::arg("component"), py::arg("x"), py::arg("y"),
      py::arg("params") = py::dict{},
      "Sample a registered analytic MHD background profile on matching arrays.");

  // -- Boundary spec --------------------------------------------------------
  // Per-side fluid/field boundary kinds (order: x_lo, x_hi, y_lo, y_hi). Names
  // pass straight through to Registry<IMhd*Boundary>::create(); the Python deck
  // validator queries registered_mhd_*_boundaries() above rather than mirroring
  // the name list, so it stays in sync with the C++ registry automatically.
  py::class_<quasar::boundary::MhdBoundarySpec>(mhd, "MhdBoundarySpec")
      .def(py::init<>())
      .def("set_fluid_all",
           [](quasar::boundary::MhdBoundarySpec& self, const std::string& kind) {
             for (int i = 0; i < 4; ++i) self.fluid[i] = kind;
           },
           py::arg("kind"))
      .def("set_fluid_side",
           [](quasar::boundary::MhdBoundarySpec& self, int side,
              const std::string& kind) {
             if (side < 0 || side > 3) throw std::out_of_range("side");
             self.fluid[side] = kind;
           },
           py::arg("side"), py::arg("kind"))
      .def("fluid_side",
           [](const quasar::boundary::MhdBoundarySpec& self, int side) {
             if (side < 0 || side > 3) throw std::out_of_range("side");
             return self.fluid[side];
           },
           py::arg("side"))
      .def("set_field_all",
           [](quasar::boundary::MhdBoundarySpec& self, const std::string& kind) {
             for (int i = 0; i < 4; ++i) self.field[i] = kind;
           },
           py::arg("kind"))
      .def("set_field_side",
           [](quasar::boundary::MhdBoundarySpec& self, int side,
              const std::string& kind) {
             if (side < 0 || side > 3) throw std::out_of_range("side");
             self.field[side] = kind;
           },
           py::arg("side"), py::arg("kind"))
      .def("field_side",
           [](const quasar::boundary::MhdBoundarySpec& self, int side) {
             if (side < 0 || side > 3) throw std::out_of_range("side");
             return self.field[side];
           },
           py::arg("side"));

  // -- Background-field spec -------------------------------------------------
  // Deck-facing static background magnetic field B0 for the field-split
  // formulation B = B0 + b. `profile` is a registry name (default "uniform");
  // bx0/by0/bz0 are the uniform-vector parameters consumed when profile ==
  // "uniform", profile_scale converts every sampled output uniformly, and
  // params carries scalar parameters for any analytic profile.
  // The Python deck validator queries
  // registered_mhd_background_profiles() above rather than mirroring the name
  // list, so it stays in sync with the C++ registry automatically.
  py::class_<quasar::mhd::MhdBackgroundSpec>(mhd, "MhdBackgroundSpec")
      .def(py::init<>())
      .def_readwrite("enabled", &quasar::mhd::MhdBackgroundSpec::enabled)
      .def_readwrite("profile", &quasar::mhd::MhdBackgroundSpec::profile)
      .def_readwrite("bx0", &quasar::mhd::MhdBackgroundSpec::bx0)
      .def_readwrite("by0", &quasar::mhd::MhdBackgroundSpec::by0)
      .def_readwrite("bz0", &quasar::mhd::MhdBackgroundSpec::bz0)
      .def_readwrite("profile_scale",
                     &quasar::mhd::MhdBackgroundSpec::profile_scale)
      .def_readwrite("params", &quasar::mhd::MhdBackgroundSpec::params)
      .def_readwrite("curl_free", &quasar::mhd::MhdBackgroundSpec::curl_free);

  // -- Config ---------------------------------------------------------------
  // Every scheme axis is a registry-name string so a new scheme is selectable
  // from Python without touching this binding (mirrors EmPicConfig).
  py::class_<quasar::mhd::MhdConfig>(mhd, "MhdConfig")
      .def(py::init<>())
      .def_readwrite("grid", &quasar::mhd::MhdConfig::grid)
      .def_readwrite("gamma", &quasar::mhd::MhdConfig::gamma)
      .def_readwrite("geometry", &quasar::mhd::MhdConfig::geometry)
      .def_readwrite("reconstruction", &quasar::mhd::MhdConfig::reconstruction)
      .def_readwrite("riemann", &quasar::mhd::MhdConfig::riemann)
      .def_readwrite("integrator", &quasar::mhd::MhdConfig::integrator)
      .def_readwrite("ct", &quasar::mhd::MhdConfig::ct)
      .def_readwrite("positivity", &quasar::mhd::MhdConfig::positivity)
      .def_readwrite("rho_floor", &quasar::mhd::MhdConfig::rho_floor)
      .def_readwrite("p_floor", &quasar::mhd::MhdConfig::p_floor)
      .def_readwrite("cfl", &quasar::mhd::MhdConfig::cfl)
      .def_readwrite("timestep_signature",
                     &quasar::mhd::MhdConfig::timestep_signature)
      .def_readwrite("boundary", &quasar::mhd::MhdConfig::boundary)
      .def_readwrite("background", &quasar::mhd::MhdConfig::background);

  // -- Solver ---------------------------------------------------------------
  py::class_<quasar::mhd::MhdSolver2D>(mhd, "MhdSolver2D")
      .def(py::init<quasar::mhd::MhdConfig>(), py::arg("config"))
      .def("step", &quasar::mhd::MhdSolver2D::step, py::arg("dt"))
      .def("step_unchecked", &quasar::mhd::MhdSolver2D::step_unchecked,
           py::arg("dt"))
      .def("advance", &quasar::mhd::MhdSolver2D::advance,
           py::arg("t_end"), py::arg("dt"))
      .def("grid", &quasar::mhd::MhdSolver2D::grid)
      .def("config", &quasar::mhd::MhdSolver2D::config,
           py::return_value_policy::reference_internal)
      .def("cfl_limit", &quasar::mhd::MhdSolver2D::cfl_limit)
      .def("divergence_b_max", &quasar::mhd::MhdSolver2D::divergence_b_max)
      .def("seed_state",
           [](quasar::mhd::MhdSolver2D& self, const std::string& component,
              const py::object& values) {
             // Stage a (storage_size,) host array into the named live-state
             // component. The array must already be in the solver's internal
             // (normalized) units and full ghost-padded storage layout.
             // Magnetic spellings accept both the staggered and the short name
             // (bx/bx_face, by/by_face, bz/bz_cell); seeding always consumes the
             // staggered internal layout. On readback, bx/by are collocated to
             // cells while bx_face/by_face expose the raw CT arrays.
             self.seed_state(component, numpy_to_real_vector(values, "values"));
           },
           py::arg("component"), py::arg("values"))
      .def("seed_background",
           [](quasar::mhd::MhdSolver2D& self, const std::string& component,
              const py::object& values) {
             // Stage a (storage_size,) host array into the named static
             // background-field component for the field-split B = B0 + b. The
             // array layout mirrors seed_state exactly (full ghost-padded
             // storage, solver-internal units). Magnetic spellings accept both
             // the staggered and the short name (b0x/b0x_face, b0y/b0y_face,
             // b0z/b0z_cell). Requires the background to be enabled.
             self.seed_background(component, numpy_to_real_vector(values, "values"));
           },
           py::arg("component"), py::arg("values"))
      .def("has_background", &quasar::mhd::MhdSolver2D::has_background)
      .def("state_component_to_host",
           [](const quasar::mhd::MhdSolver2D& self, const std::string& component) {
             return vector_to_numpy(self.state_component_to_host(component));
           },
           py::arg("component"),
           "Read one storage-sized state array. 'bx'/'by' use the same "
           "finite-volume face-to-cell collocation as the EOS; 'bx_face'/"
           "'by_face' return the raw staggered CT samples.");
}
