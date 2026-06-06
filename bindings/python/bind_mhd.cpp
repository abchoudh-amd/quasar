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

#include "quasar/boundary/mhd_boundary.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/numerics/ct_scheme.hpp"
#include "quasar/numerics/flux_reconstruction.hpp"
#include "quasar/numerics/positivity_limiter.hpp"
#include "quasar/numerics/riemann_solver.hpp"
#include "quasar/numerics/ssprk_integrator.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using ::quasar::Real;

// Forces a contiguous, correctly-typed array at the binding boundary so the raw
// data()+shape(0) copy below never reads strided/wrong-type memory (mirrors the
// RealArray alias in bind_pic.cpp).
using RealArray = py::array_t<Real, py::array::c_style | py::array::forcecast>;

std::vector<Real> numpy_to_real_vector(const RealArray& arr, const char* what) {
  if (arr.ndim() != 1) {
    throw std::invalid_argument(std::string{what} + ": expected 1-D NumPy array");
  }
  const auto* data = arr.data();
  return std::vector<Real>{data, data + arr.shape(0)};
}

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
      m.def_submodule("mhd", "2D ideal-MHD (finite-difference + CT) scaffolding.");

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
      .def_readwrite("boundary", &quasar::mhd::MhdConfig::boundary);

  // -- Solver ---------------------------------------------------------------
  py::class_<quasar::mhd::MhdSolver2D>(mhd, "MhdSolver2D")
      .def(py::init<quasar::mhd::MhdConfig>(), py::arg("config"))
      .def("step", &quasar::mhd::MhdSolver2D::step, py::arg("dt"))
      .def("advance", &quasar::mhd::MhdSolver2D::advance,
           py::arg("t_end"), py::arg("dt"))
      .def("grid", &quasar::mhd::MhdSolver2D::grid)
      .def("config", &quasar::mhd::MhdSolver2D::config,
           py::return_value_policy::reference_internal)
      .def("cfl_limit", &quasar::mhd::MhdSolver2D::cfl_limit)
      .def("divergence_b_max", &quasar::mhd::MhdSolver2D::divergence_b_max)
      .def("seed_state",
           [](quasar::mhd::MhdSolver2D& self, const std::string& component,
              const RealArray& values) {
             // Stage a (storage_size,) host array into the named live-state
             // component. The array must already be in the solver's internal
             // (normalized) units and full ghost-padded storage layout
             // (matches state_component_to_host()). Magnetic spellings accept
             // both the staggered and the short name (bx/bx_face, by/by_face,
             // bz/bz_cell); the solver resolves the alias.
             self.seed_state(component, numpy_to_real_vector(values, "values"));
           },
           py::arg("component"), py::arg("values"))
      .def("state_component_to_host",
           [](const quasar::mhd::MhdSolver2D& self, const std::string& component) {
             return vector_to_numpy(self.state_component_to_host(component));
           },
           py::arg("component"));
}
