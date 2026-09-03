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
#include "quasar/core/field_source.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/numerics/ct_scheme.hpp"
#include "quasar/numerics/field_evaluator.hpp"
#include "quasar/numerics/flux_reconstruction.hpp"
#include "quasar/numerics/mhd_background_profile.hpp"
#include "quasar/numerics/positivity_limiter.hpp"
#include "quasar/numerics/riemann_solver.hpp"
#include "quasar/numerics/ssprk_integrator.hpp"
#include "quasar/physics/mhd/background_builder.hpp"
#include "quasar/physics/mhd/initial_conditions.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using ::quasar::Real;
using ::quasar::python_detail::numpy_to_real_vector;
using ::quasar::python_detail::require_real_array;

// Fixed-size C arrays inside a bound struct cannot use def_readwrite: pybind
// would hand out a pointer with no length. Expose each as a list property of
// exactly the declared arity instead, so a wrong-length deck value is a
// ValueError at the binding rather than a silent partial write.
template <std::size_t N>
py::list array_to_list(const Real (&values)[N]) {
  py::list out;
  for (std::size_t k = 0; k < N; ++k) out.append(values[k]);
  return out;
}

template <std::size_t N>
void list_to_array(Real (&values)[N], const py::sequence& source,
                   const char* what) {
  if (static_cast<std::size_t>(py::len(source)) != N) {
    throw std::invalid_argument{std::string{what} + " requires exactly " +
                                std::to_string(N) + " components"};
  }
  for (std::size_t k = 0; k < N; ++k) {
    values[k] = py::cast<Real>(source[k]);
  }
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
  // Halo width and spatial order for a reconstruction name, resolved through the
  // registry and numerics::reconstruction_order_from_nghost. The Python deck and
  // distributed runner need both BEFORE a solver exists (they build the padded
  // initial state that the solver is then seeded from), so they cannot read
  // solver.grid().nghost. Exposing the C++ mapping keeps that pre-solver path on
  // the same single source of truth as the solver itself instead of a mirrored
  // Python table that must be edited in lockstep when a scheme is added.
  //
  // Registry::create throws std::out_of_range on an unknown name, which pybind
  // surfaces as IndexError -- a misleading type for a bad scheme string, and the
  // dict lookups these bindings replace raised KeyError. Translate to
  // invalid_argument (ValueError) and name the offending scheme.
  auto reconstruction_nghost = [](const std::string& name) {
    try {
      return quasar::Registry<quasar::numerics::IFluxReconstruction>::instance()
          .create(name)
          ->required_nghost();
    } catch (const std::out_of_range&) {
      throw std::invalid_argument{
          "unknown MHD reconstruction scheme '" + name + "'"};
    }
  };
  mhd.def("reconstruction_halo", reconstruction_nghost, py::arg("name"),
          "Ghost-cell halo required by the named flux-reconstruction scheme.");
  mhd.def("reconstruction_order",
          [reconstruction_nghost](const std::string& name) {
            return quasar::numerics::reconstruction_order_from_nghost(
                reconstruction_nghost(name));
          },
          py::arg("name"),
          "Spatial order of the device kernel selected by the named scheme.");
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
  // Registry introspection, not a deck path. The deck's background builder
  // lowers a profile to an affine POD and samples it in a kernel
  // (physics/mhd/background_builder.hpp); this entry point exists so a test or
  // a notebook can ask a registered profile what it returns, which is exactly
  // the host virtual the lowering probes.
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

  // -- Seeded initial state --------------------------------------------------
  // The deck still selects a generator by string; `kind` accepts and returns
  // that name, and registered_initial_conditions() is the list the Python deck
  // validator checks against rather than mirroring it.
  mhd.def("registered_initial_conditions",
          &quasar::mhd::registered_mhd_initial_conditions,
          "Names of the built-in MHD initial-condition generators.");

  using quasar::mhd::MhdInitialConditionSpec;
  py::class_<MhdInitialConditionSpec>(mhd, "MhdInitialConditionSpec")
      .def(py::init<>())
      .def_property(
          "kind",
          [](const MhdInitialConditionSpec& s) {
            return quasar::mhd::registered_mhd_initial_conditions()
                .at(static_cast<std::size_t>(s.kind));
          },
          [](MhdInitialConditionSpec& s, const std::string& name) {
            s.kind = quasar::mhd::initial_condition_kind(name);
          })
      .def_readwrite("grid", &MhdInitialConditionSpec::grid)
      .def_readwrite("gamma", &MhdInitialConditionSpec::gamma)
      .def_readwrite("cylindrical", &MhdInitialConditionSpec::cylindrical)
      .def_readwrite("scheme_order", &MhdInitialConditionSpec::scheme_order)
      .def_readwrite("magnetic_scale", &MhdInitialConditionSpec::magnetic_scale)
      .def_readwrite("interface", &MhdInitialConditionSpec::interface)
      .def_property(
          "left",
          [](const MhdInitialConditionSpec& s) { return array_to_list(s.left); },
          [](MhdInitialConditionSpec& s, const py::sequence& v) {
            list_to_array(s.left, v, "left state");
          })
      .def_property(
          "right",
          [](const MhdInitialConditionSpec& s) { return array_to_list(s.right); },
          [](MhdInitialConditionSpec& s, const py::sequence& v) {
            list_to_array(s.right, v, "right state");
          })
      .def_readwrite("rho", &MhdInitialConditionSpec::rho)
      .def_readwrite("pressure", &MhdInitialConditionSpec::pressure)
      .def_readwrite("b0", &MhdInitialConditionSpec::b0)
      .def_readwrite("total_b0", &MhdInitialConditionSpec::total_b0)
      .def_readwrite("amplitude", &MhdInitialConditionSpec::amplitude)
      .def_readwrite("wavenumber", &MhdInitialConditionSpec::wavenumber)
      .def_readwrite("magnetic_velocity_scale",
                     &MhdInitialConditionSpec::magnetic_velocity_scale)
      .def_property(
          "b_uniform",
          [](const MhdInitialConditionSpec& s) {
            return array_to_list(s.b_uniform);
          },
          [](MhdInitialConditionSpec& s, const py::sequence& v) {
            list_to_array(s.b_uniform, v, "uniform background field");
          })
      .def_property(
          "center",
          [](const MhdInitialConditionSpec& s) {
            return array_to_list(s.center);
          },
          [](MhdInitialConditionSpec& s, const py::sequence& v) {
            list_to_array(s.center, v, "center");
          })
      .def_readwrite("r_in", &MhdInitialConditionSpec::r_in)
      .def_readwrite("r0", &MhdInitialConditionSpec::r0)
      .def_readwrite("r1", &MhdInitialConditionSpec::r1)
      .def_readwrite("rho_in", &MhdInitialConditionSpec::rho_in)
      .def_readwrite("rho_out", &MhdInitialConditionSpec::rho_out)
      .def_readwrite("p_in", &MhdInitialConditionSpec::p_in)
      .def_readwrite("p_out", &MhdInitialConditionSpec::p_out)
      .def_readwrite("p_core", &MhdInitialConditionSpec::p_core)
      .def_readwrite("p_ambient", &MhdInitialConditionSpec::p_ambient)
      .def_readwrite("rho_ambient", &MhdInitialConditionSpec::rho_ambient)
      .def_readwrite("u0", &MhdInitialConditionSpec::u0)
      .def_readwrite("blob_half", &MhdInitialConditionSpec::blob_half);

  mhd.def(
      "build_initial_state",
      [](const MhdInitialConditionSpec& spec) {
        quasar::mhd::MhdField2D<Real> state{spec.grid};
        quasar::mhd::build_initial_state(spec, state);
        // Download at the output boundary and nowhere earlier. The distributed
        // runner slices these padded arrays per tile and the deck tests compare
        // them to closed-form references, so a host copy is the product here --
        // but every value in it was computed on device.
        py::dict out;
        auto emit = [&](const char* name,
                        const quasar::backend::DeviceBuffer<Real>& buffer) {
          std::vector<Real> host(buffer.size());
          buffer.copy_to_host(host.data(), host.size());
          out[name] = vector_to_numpy(host);
        };
        emit("rho", state.rho);
        emit("mx", state.mx);
        emit("my", state.my);
        emit("mz", state.mz);
        emit("energy", state.energy);
        emit("bx", state.bx_face);
        emit("by", state.by_face);
        emit("bz", state.bz_cell);
        return out;
      },
      py::arg("spec"),
      "Build the ghost-padded conserved initial state on device and return it "
      "as one flat host array per STATE_COMPONENT, in solver-internal units.");

  // -- Background construction -----------------------------------------------
  // Four deck sources, one validated product. Every one of them assembles B0 in
  // device memory, proves it discretely solenoidal and boundary-compatible
  // there, and only then downloads -- the host arrays returned are the output
  // boundary, not an intermediate.
  using quasar::mhd::MhdBackgroundBuildSpec;
  py::class_<MhdBackgroundBuildSpec>(mhd, "MhdBackgroundBuildSpec")
      .def(py::init<>())
      .def_readwrite("grid", &MhdBackgroundBuildSpec::grid)
      .def_readwrite("cylindrical", &MhdBackgroundBuildSpec::cylindrical)
      .def_readwrite("magnetic_scale", &MhdBackgroundBuildSpec::magnetic_scale)
      .def_readwrite("b_scale", &MhdBackgroundBuildSpec::b_scale)
      .def_readwrite("bz0", &MhdBackgroundBuildSpec::bz0)
      .def_readwrite("vacuum_project", &MhdBackgroundBuildSpec::vacuum_project)
      .def_property(
          "field_modes",
          [](const MhdBackgroundBuildSpec& s) {
            py::list out;
            for (int k = 0; k < 4; ++k) out.append(s.field_modes[k]);
            return out;
          },
          [](MhdBackgroundBuildSpec& s, const py::sequence& v) {
            if (py::len(v) != 4) {
              throw std::invalid_argument{
                  "field_modes requires exactly four side codes"};
            }
            for (int k = 0; k < 4; ++k) {
              s.field_modes[k] = py::cast<int>(v[k]);
            }
          })
      .def(
          "set_profile",
          [](MhdBackgroundBuildSpec& self, const std::string& name,
             const py::dict& params) {
            // Configure the registered profile, then lower it. The registry
            // object stays the single definition of what the name means; see
            // physics/mhd/background_builder.hpp for why the lowering is affine
            // and why a nonlinear profile is refused rather than approximated.
            std::unique_ptr<quasar::numerics::IMhdBackgroundProfile> profile;
            try {
              profile = quasar::Registry<
                  quasar::numerics::IMhdBackgroundProfile>::instance()
                  .create(name);
            } catch (const std::out_of_range&) {
              throw std::invalid_argument{
                  "unknown MHD background profile '" + name + "'"};
            }
            for (const auto& item : params) {
              const std::string key = py::cast<std::string>(item.first);
              const Real value = py::cast<Real>(item.second);
              if (!std::isfinite(value)) {
                throw std::invalid_argument{"background parameter " + key +
                                            " must be finite"};
              }
              if (!profile->set_parameter(key, value)) {
                throw std::invalid_argument{
                    "unknown parameter " + key + " for background profile " +
                    name};
              }
            }
            self.profile =
                quasar::mhd::lower_affine_background_profile(*profile, name);
          },
          py::arg("name"), py::arg("params") = py::dict{});

  auto download_background =
      [](const quasar::mhd::MhdBackgroundField<Real>& b0) {
        py::dict out;
        auto emit = [&](const char* name,
                        const quasar::backend::DeviceBuffer<Real>& buffer) {
          std::vector<Real> host(buffer.size());
          buffer.copy_to_host(host.data(), host.size());
          out[name] = vector_to_numpy(host);
        };
        emit("b0x", b0.b0x_face);
        emit("b0y", b0.b0y_face);
        emit("b0z", b0.b0z_cell);
        return out;
      };

  mhd.def(
      "build_background_from_profile",
      [download_background](const MhdBackgroundBuildSpec& spec) {
        quasar::mhd::MhdBackgroundField<Real> b0{spec.grid};
        quasar::mhd::build_background_from_profile(spec, b0);
        quasar::mhd::validate_deck_background(spec, b0);
        return download_background(b0);
      },
      py::arg("spec"),
      "Sample the configured analytic profile on the staggered padded mesh.");

  mhd.def(
      "build_background_from_corner_potential",
      [download_background](const MhdBackgroundBuildSpec& spec,
                            const py::object& values) {
        // The npz is read in Python because loading a file is not calculation.
        // From this upload onward nothing leaves the device until the result is
        // validated.
        const auto host = numpy_to_real_vector(values, "corner potential");
        quasar::backend::DeviceBuffer<Real> a_corners(host.size());
        a_corners.copy_from_host(host.data(), host.size());
        quasar::mhd::MhdBackgroundField<Real> b0{spec.grid};
        quasar::mhd::build_background_from_corner_potential(spec, a_corners, b0);
        quasar::mhd::validate_deck_background(spec, b0);
        return download_background(b0);
      },
      py::arg("spec"), py::arg("a_corners"),
      "Curl a lab-Y corner vector potential, flattened row-major over the "
      "padded (height+1) x (pitch+1) corner grid.");

  mhd.def(
      "build_background_from_conductors",
      [download_background](const MhdBackgroundBuildSpec& spec,
                            const quasar::core::IFieldSource& conductors,
                            const quasar::numerics::IFieldEvaluator& evaluator) {
        quasar::mhd::MhdBackgroundField<Real> b0{spec.grid};
        quasar::mhd::build_background_from_conductors(spec, conductors,
                                                      evaluator, b0);
        quasar::mhd::validate_deck_background(spec, b0);
        return download_background(b0);
      },
      py::arg("spec"), py::arg("conductors"), py::arg("evaluator"),
      "Evaluate the inline coil geometry on the padded corner grid and curl "
      "the result, with no host round trip between the two.");

  mhd.def(
      "build_background_from_arrays",
      [download_background](const MhdBackgroundBuildSpec& spec,
                            const py::object& b0x, const py::object& b0y,
                            const py::object& b0z) {
        quasar::mhd::MhdBackgroundField<Real> b0{spec.grid};
        auto stage = [&](const py::object& values, const char* what,
                         quasar::backend::DeviceBuffer<Real>& buffer) {
          const auto host = numpy_to_real_vector(values, what);
          if (host.size() != buffer.size()) {
            throw std::invalid_argument{
                std::string{what} +
                " does not match the padded background storage size"};
          }
          buffer.copy_from_host(host.data(), host.size());
        };
        stage(b0x, "b0x", b0.b0x_face);
        stage(b0y, "b0y", b0.b0y_face);
        stage(b0z, "b0z", b0.b0z_cell);
        quasar::mhd::scale_explicit_background(spec, b0);
        quasar::mhd::validate_deck_background(spec, b0);
        return download_background(b0);
      },
      py::arg("spec"), py::arg("b0x"), py::arg("b0y"), py::arg("b0z"),
      "Unit-convert and validate an explicit deck-supplied background field.");

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
      // Returned as a plain (mass, energy) tuple rather than a bound struct:
      // the CLI's only use is to write two floats into the output npz, and a
      // one-field-per-attribute wrapper class would be pure ceremony. Note this
      // is NOT an output-boundary conversion the caller may redo -- the
      // cylindrical volume weight is applied inside the reduction, so summing
      // the downloaded planes in NumPy instead would give a different number.
      .def("conserved_cell_totals",
           [](const quasar::mhd::MhdSolver2D& self) {
             const auto totals = self.conserved_cell_totals();
             return pybind11::make_tuple(totals.mass, totals.energy);
           })
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
