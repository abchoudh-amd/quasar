// Pybind11 bindings for the Quasar PIC module.
//
// Exposes the 2D-3V PIC scaffolding plus the surface needed to drive a full
// simulation from Python: species construction + host particle upload,
// external-field sampling against a ConductorSystem (reused from the
// magnetostatics bindings), Yee field readback as NumPy arrays, and the
// diagnostics declared in physics/pic/diagnostics.hpp.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numpy_utils.hpp"

#include "quasar/backend/memory.hpp"
#include "quasar/boundary/boundary_condition.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/normalization.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/filter.hpp"
#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/pic/diagnostics.hpp"
#include "quasar/physics/pic/particle_sampling.hpp"
#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/physics/pic/species.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using ::quasar::Real;
using ::quasar::python_detail::numpy_to_finite_real_vector;
using ::quasar::python_detail::numpy_to_real_vector;

py::array_t<Real> buffer_to_numpy(const quasar::backend::DeviceBuffer<Real>& buf) {
  const py::ssize_t n = static_cast<py::ssize_t>(buf.size());
  py::array_t<Real> arr(n);
  if (n > 0) buf.copy_to_host(arr.mutable_data(), buf.size());
  return arr;
}

// Single source of truth for the component-name -> Yee buffer mapping. Works
// for const and mutable field refs alike; keep the accepted names in sync with
// FIELD_COMPONENTS in python/quasar/pic/io.py.
template <typename FieldRef>
auto& yee_component_buffer(FieldRef& f, const std::string& component) {
  if (component == "ex") return f.ex;
  if (component == "ey") return f.ey;
  if (component == "ez") return f.ez;
  if (component == "bx") return f.bx;
  if (component == "by") return f.by;
  if (component == "bz") return f.bz;
  throw std::invalid_argument("unknown Yee field component '" + component + "'");
}

py::array_t<Real> yee_component_to_numpy(const quasar::YeeField2D<Real>& f,
                                         const std::string& component) {
  return buffer_to_numpy(yee_component_buffer(f, component));
}

py::dict yee_field_to_dict(const quasar::YeeField2D<Real>& f) {
  py::dict d;
  d["ex"] = buffer_to_numpy(f.ex);
  d["ey"] = buffer_to_numpy(f.ey);
  d["ez"] = buffer_to_numpy(f.ez);
  d["bx"] = buffer_to_numpy(f.bx);
  d["by"] = buffer_to_numpy(f.by);
  d["bz"] = buffer_to_numpy(f.bz);
  d["nx"] = f.grid.nx;
  d["ny"] = f.grid.ny;
  return d;
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

void bind_pic(py::module_& m) {
  py::module_ pic = m.def_submodule("pic", "2D3V electromagnetic PIC scaffolding.");

  // -- Grid / normalization / config ---------------------------------------

  py::class_<quasar::Grid2D>(pic, "Grid2D")
      .def(py::init<int, int, Real, Real, Real, Real, int>(),
           py::arg("nx"), py::arg("ny"), py::arg("lx"), py::arg("ly"),
           py::arg("origin_x") = 0.0, py::arg("origin_y") = 0.0,
           py::arg("nghost") = 1)
      .def_readwrite("nx", &quasar::Grid2D::nx)
      .def_readwrite("ny", &quasar::Grid2D::ny)
      .def_readwrite("lx", &quasar::Grid2D::lx)
      .def_readwrite("ly", &quasar::Grid2D::ly)
      .def_readwrite("origin_x", &quasar::Grid2D::origin_x)
      .def_readwrite("origin_y", &quasar::Grid2D::origin_y)
      .def_readwrite("nghost", &quasar::Grid2D::nghost)
      .def("dx", &quasar::Grid2D::dx)
      .def("dy", &quasar::Grid2D::dy);

  pic.def("required_nghost", &quasar::required_nghost, py::arg("fdtd_order"),
          "Minimum ghost-cell halo for the given FDTD order (1 for order 2, 2 for order 4).");

  // Registry introspection: expose the registered plugin names so the Python
  // deck validators select against the live C++ registry instead of a hardcoded
  // mirror. Adding a boundary/filter via QUASAR_REGISTER_* then needs no Python
  // edit. Sorted for stable error messages.
  auto sorted_names = [](std::vector<std::string> v) {
    // "internal" is a runtime-only tile interface, never physics-deck
    // vocabulary.  Keep registry introspection authoritative for public
    // plug-ins without allowing users to request a no-op exterior boundary.
    std::erase(v, "internal");
    std::sort(v.begin(), v.end());
    return v;
  };
  pic.def("registered_particle_boundaries",
          [sorted_names]() {
            return sorted_names(
                quasar::Registry<quasar::boundary::IParticleBoundary>::instance().names());
          },
          "Names of registered particle boundary conditions.");
  pic.def("registered_field_boundaries",
          [sorted_names]() {
            return sorted_names(
                quasar::Registry<quasar::boundary::IFieldBoundary>::instance().names());
          },
          "Names of registered field boundary conditions.");
  pic.def("registered_current_filters",
          [sorted_names]() {
            return sorted_names(
                quasar::Registry<quasar::numerics::ICurrentFilter>::instance().names());
          },
          "Names of registered current-smoothing filters.");

  py::enum_<quasar::UnitTag>(pic, "UnitTag")
      .value("time", quasar::UnitTag::time)
      .value("length", quasar::UnitTag::length)
      .value("velocity", quasar::UnitTag::velocity)
      .value("e_field", quasar::UnitTag::e_field)
      .value("b_field", quasar::UnitTag::b_field)
      .value("density", quasar::UnitTag::density)
      .value("charge", quasar::UnitTag::charge)
      .value("mass", quasar::UnitTag::mass)
      .value("temperature_eV", quasar::UnitTag::temperature_eV);

  py::class_<quasar::Normalization>(pic, "Normalization")
      .def(py::init<>())
      .def_static("identity", &quasar::Normalization::identity)
      .def_static("plasma", &quasar::Normalization::plasma,
                  py::arg("n_ref"), py::arg("q_ref"), py::arg("m_ref"))
      .def("is_identity", &quasar::Normalization::is_identity)
      .def("to_internal", &quasar::Normalization::to_internal,
           py::arg("value"), py::arg("tag"))
      .def("to_si", &quasar::Normalization::to_si,
           py::arg("value"), py::arg("tag"))
      .def("length_scale", &quasar::Normalization::length_scale)
      .def("time_scale", &quasar::Normalization::time_scale)
      .def("e_field_scale", &quasar::Normalization::e_field_scale)
      .def("b_field_scale", &quasar::Normalization::b_field_scale)
      .def("temperature_eV_scale", &quasar::Normalization::temperature_eV_scale)
      .def_readonly("n_ref", &quasar::Normalization::n_ref)
      .def_readonly("q_ref", &quasar::Normalization::q_ref)
      .def_readonly("m_ref", &quasar::Normalization::m_ref)
      .def_readonly("omega_p_ref", &quasar::Normalization::omega_p_ref);

  // Boundaries are selected by registry name (no enum): the string is passed
  // straight through to Registry<I*Boundary>::create(). Adding a boundary is a
  // single QUASAR_REGISTER_*_BOUNDARY with nothing to update here — the Python
  // deck validator queries registered_*_boundaries() above rather than mirroring
  // the name list, so it stays in sync automatically.
  py::class_<quasar::boundary::BoundarySpec>(pic, "BoundarySpec")
      .def(py::init<>())
      .def("set_particle_all",
           [](quasar::boundary::BoundarySpec& self, const std::string& kind) {
             for (int i = 0; i < 4; ++i) self.particle[i] = kind;
           },
           py::arg("kind"))
      .def("set_particle_side",
           [](quasar::boundary::BoundarySpec& self, int side,
              const std::string& kind) {
             if (side < 0 || side > 3) throw std::out_of_range("side");
             self.particle[side] = kind;
           },
           py::arg("side"), py::arg("kind"))
      .def("particle_side",
           [](const quasar::boundary::BoundarySpec& self, int side) {
             if (side < 0 || side > 3) throw std::out_of_range("side");
             return self.particle[side];
           },
           py::arg("side"))
      .def("set_field_side",
           [](quasar::boundary::BoundarySpec& self, int side,
              const std::string& kind) {
             if (side < 0 || side > 3) throw std::out_of_range("side");
             self.field[side] = kind;
           },
           py::arg("side"), py::arg("kind"))
      .def("field_side",
           [](const quasar::boundary::BoundarySpec& self, int side) {
             if (side < 0 || side > 3) throw std::out_of_range("side");
             return self.field[side];
           },
           py::arg("side"));

  py::class_<quasar::pic::FilterSpec>(pic, "FilterSpec")
      .def(py::init([](std::string name, int passes) {
             return quasar::pic::FilterSpec{std::move(name), passes};
           }),
           py::arg("name"), py::arg("passes") = 1)
      .def_readwrite("name", &quasar::pic::FilterSpec::name)
      .def_readwrite("passes", &quasar::pic::FilterSpec::passes);

  py::class_<quasar::pic::EmPicConfig>(pic, "EmPicConfig")
      .def(py::init<>())
      .def_readwrite("grid", &quasar::pic::EmPicConfig::grid)
      .def_readwrite("fdtd_order", &quasar::pic::EmPicConfig::fdtd_order)
      .def_readwrite("shape", &quasar::pic::EmPicConfig::shape)
      .def_readwrite("plane", &quasar::pic::EmPicConfig::plane)
      .def_readwrite("geometry", &quasar::pic::EmPicConfig::geometry)
      .def_readwrite("boundary", &quasar::pic::EmPicConfig::boundary)
      .def_readwrite("normalization", &quasar::pic::EmPicConfig::normalization)
      .def_readwrite("filters", &quasar::pic::EmPicConfig::filters)
      .def_readwrite("neutralizing_background",
                     &quasar::pic::EmPicConfig::neutralizing_background)
      .def_readwrite("external_field_signature",
                     &quasar::pic::EmPicConfig::external_field_signature)
      .def_readwrite("timestep_signature",
                     &quasar::pic::EmPicConfig::timestep_signature);

  // -- Species --------------------------------------------------------------

  py::class_<quasar::pic::SpeciesConfig>(pic, "SpeciesConfig")
      .def(py::init([](std::string name, Real charge, Real mass, std::size_t cap) {
             return quasar::pic::SpeciesConfig{std::move(name), charge, mass, cap};
           }),
           py::arg("name"), py::arg("charge"), py::arg("mass"),
           py::arg("capacity"))
      .def_readwrite("name", &quasar::pic::SpeciesConfig::name)
      .def_readwrite("charge", &quasar::pic::SpeciesConfig::charge)
      .def_readwrite("mass", &quasar::pic::SpeciesConfig::mass)
      .def_readwrite("capacity", &quasar::pic::SpeciesConfig::capacity);

  // Keyword-only construction with defaults, so a deck that uses none of the
  // optional perturbation fields does not have to name them.
  py::class_<quasar::pic::ParticleSampleConfig>(pic, "ParticleSampleConfig")
      .def(py::init<>())
      .def_readwrite("count", &quasar::pic::ParticleSampleConfig::count)
      .def_readwrite("x_min", &quasar::pic::ParticleSampleConfig::x_min)
      .def_readwrite("x_max", &quasar::pic::ParticleSampleConfig::x_max)
      .def_readwrite("y_min", &quasar::pic::ParticleSampleConfig::y_min)
      .def_readwrite("y_max", &quasar::pic::ParticleSampleConfig::y_max)
      .def_readwrite("cylindrical",
                     &quasar::pic::ParticleSampleConfig::cylindrical)
      .def_readwrite("thermal_speed",
                     &quasar::pic::ParticleSampleConfig::thermal_speed)
      .def_readwrite("drift_x", &quasar::pic::ParticleSampleConfig::drift_x)
      .def_readwrite("drift_y", &quasar::pic::ParticleSampleConfig::drift_y)
      .def_readwrite("drift_z", &quasar::pic::ParticleSampleConfig::drift_z)
      .def_readwrite("seed", &quasar::pic::ParticleSampleConfig::seed)
      .def_readwrite("species_key",
                     &quasar::pic::ParticleSampleConfig::species_key)
      .def_readwrite("perturb", &quasar::pic::ParticleSampleConfig::perturb)
      .def_readwrite("mode_x", &quasar::pic::ParticleSampleConfig::mode_x)
      .def_readwrite("mode_y", &quasar::pic::ParticleSampleConfig::mode_y)
      .def_readwrite("phase", &quasar::pic::ParticleSampleConfig::phase)
      .def_readwrite("amplitude_x",
                     &quasar::pic::ParticleSampleConfig::amplitude_x)
      .def_readwrite("amplitude_y",
                     &quasar::pic::ParticleSampleConfig::amplitude_y)
      .def_readwrite("amplitude_z",
                     &quasar::pic::ParticleSampleConfig::amplitude_z)
      .def_readwrite("domain_origin_x",
                     &quasar::pic::ParticleSampleConfig::domain_origin_x)
      .def_readwrite("domain_origin_y",
                     &quasar::pic::ParticleSampleConfig::domain_origin_y)
      .def_readwrite("domain_lx",
                     &quasar::pic::ParticleSampleConfig::domain_lx)
      .def_readwrite("domain_ly",
                     &quasar::pic::ParticleSampleConfig::domain_ly)
      .def_readwrite("weight", &quasar::pic::ParticleSampleConfig::weight);

  // The per-particle block measure and the lattice stride are O(1) scalars the
  // deck layer needs before it can form a macro weight, so they are exposed
  // rather than recomputed in Python.
  pic.def("quiet_block_measure", &quasar::pic::quiet_block_measure,
          py::arg("config"),
          "Area (Cartesian) or ring volume (cylindrical) per equal-weight "
          "particle.");
  // Standalone sampling into host arrays, for the callers that genuinely need
  // the particles on the host: the distributed runner partitions the sample
  // across ranks before any solver exists. It runs the same kernels as
  // EmPic2D3V::sample_species_particles, so a serial and a distributed run of
  // one deck draw the same velocities.
  pic.def("sample_particles",
          [](const quasar::pic::ParticleSampleConfig& config) {
            quasar::pic::SpeciesConfig species_config;
            species_config.name = "sample";
            species_config.capacity = config.count;
            quasar::pic::ParticleSpecies species{species_config};
            quasar::pic::sample_species(species, config, nullptr);
            const auto snapshot = species.to_host();
            py::dict out;
            out["x"] = py::array_t<Real>(snapshot.x.size(), snapshot.x.data());
            out["y"] = py::array_t<Real>(snapshot.y.size(), snapshot.y.data());
            out["vx"] =
                py::array_t<Real>(snapshot.vx.size(), snapshot.vx.data());
            out["vy"] =
                py::array_t<Real>(snapshot.vy.size(), snapshot.vy.data());
            out["vz"] =
                py::array_t<Real>(snapshot.vz.size(), snapshot.vz.data());
            out["weight"] = py::array_t<Real>(snapshot.weight.size(),
                                              snapshot.weight.data());
            return out;
          },
          py::arg("config"),
          "Sample one species' quiet-start particles and return them as host "
          "arrays. Same kernels as EmPic2D3V.sample_species_particles.");

  pic.def("quiet_start_stride", &quasar::pic::quiet_start_stride,
          py::arg("count"),
          "Rank-1 lattice stride: the golden-ratio multiple of count raised "
          "to coprimality with it.");

  // NOTE: ParticleSpecies is move-only (DeviceBuffer is non-copyable). Python
  // users never construct or mutate one directly. The solver owns uploads so it
  // can validate the physical domain and invalidate charge/background caches;
  // species_at(idx) below is a read-only snapshot/count handle.
  py::class_<quasar::pic::ParticleSpecies>(pic, "ParticleSpecies")
      .def_property_readonly("name", &quasar::pic::ParticleSpecies::name)
      .def_property_readonly("charge", &quasar::pic::ParticleSpecies::charge)
      .def_property_readonly("mass", &quasar::pic::ParticleSpecies::mass)
      .def_property_readonly("size", &quasar::pic::ParticleSpecies::size)
      .def_property_readonly("capacity", &quasar::pic::ParticleSpecies::capacity)
      .def("to_host",
           [](const quasar::pic::ParticleSpecies& self) {
             auto snap = self.to_host();
             // Preserve the established serial binding schema. Distributed
             // migration/checkpoint state is exposed by bind_distributed.cpp's
             // dedicated full-state converter, not this low-level snapshot.
             py::dict d;
             d["x"] = vector_to_numpy(snap.x);
             d["y"] = vector_to_numpy(snap.y);
             d["vx"] = vector_to_numpy(snap.vx);
             d["vy"] = vector_to_numpy(snap.vy);
             d["vz"] = vector_to_numpy(snap.vz);
             d["weight"] = vector_to_numpy(snap.weight);
             d["alive"] = vector_to_numpy(snap.alive);
             return d;
           });

  // -- Solver ---------------------------------------------------------------

  py::class_<quasar::pic::EmPic2D3V>(pic, "EmPic2D3V")
      .def(py::init<quasar::pic::EmPicConfig>(), py::arg("config"))
      .def("step", &quasar::pic::EmPic2D3V::step, py::arg("dt"))
      .def("advance", &quasar::pic::EmPic2D3V::advance,
           py::arg("t_end"), py::arg("dt"))
      .def("finalize", &quasar::pic::EmPic2D3V::finalize)
      .def("add_species",
           [](quasar::pic::EmPic2D3V& self,
              const quasar::pic::SpeciesConfig& cfg) {
             self.add_species(quasar::pic::ParticleSpecies{cfg});
             return self.species().size() - 1;  // index of the new species
           },
           py::arg("config"))
      .def("set_species_particles",
           [](quasar::pic::EmPic2D3V& self, std::size_t index,
              const py::object& x, const py::object& y,
              const py::object& vx, const py::object& vy,
              const py::object& vz, const py::object& w) {
             self.set_species_particles(
                 index, numpy_to_real_vector(x, "x"),
                 numpy_to_real_vector(y, "y"),
                 numpy_to_real_vector(vx, "vx"),
                 numpy_to_real_vector(vy, "vy"),
                 numpy_to_real_vector(vz, "vz"),
                 numpy_to_real_vector(w, "weight"));
           },
           py::arg("index"), py::arg("x"), py::arg("y"),
           py::arg("vx"), py::arg("vy"), py::arg("vz"),
           py::arg("weight"),
           "Upload physical particle positions and velocities at t=0. "
           "Velocity must not be pre-staggered; the first step applies the "
           "initial half-width Boris force update before drifting.")
      .def("sample_species_particles",
           [](quasar::pic::EmPic2D3V& self, std::size_t index,
              const quasar::pic::ParticleSampleConfig& config) {
             self.sample_species_particles(index, config);
           },
           py::arg("index"), py::arg("config"),
           "Sample this species' quiet-start positions and Maxwellian "
           "velocities directly into device memory. Replaces building the "
           "arrays on the host and calling set_species_particles; the "
           "velocities come from a counter-based Philox generator and are "
           "therefore different draws than the NumPy path produced.")
      .def("species_count",
           [](const quasar::pic::EmPic2D3V& self) { return self.species().size(); })
      .def("species_alive_count",
           [](quasar::pic::EmPic2D3V& self, std::size_t idx) {
             if (idx >= self.species().size()) {
               throw std::out_of_range("species_alive_count: index out of range");
             }
             return quasar::pic::alive_count(self.species()[idx]);
           },
           py::arg("index"))
      .def("species_at",
           [](const quasar::pic::EmPic2D3V& self, std::size_t idx)
               -> const quasar::pic::ParticleSpecies& {
             if (idx >= self.species().size()) {
               throw std::out_of_range("species_at: index out of range");
             }
             return self.species()[idx];
           },
           py::return_value_policy::reference_internal, py::arg("index"))
      .def("sample_external_field",
           [](quasar::pic::EmPic2D3V& self,
              quasar::numerics::IFieldEvaluator& evaluator,
              const quasar::magnetostatics::ConductorSystem& conductors,
              Real length_scale, Real e_field_scale, Real b_field_scale,
              const std::string& plane) {
             quasar::pic::sample_external_field(evaluator, conductors,
                                                self.external_fields(),
                                                length_scale, e_field_scale,
                                                b_field_scale, plane,
                                                self.config().geometry,
                                                self.config().fdtd_order);
           },
           py::arg("evaluator"), py::arg("conductors"),
           py::arg("length_scale") = 1.0, py::arg("e_field_scale") = 1.0,
           py::arg("b_field_scale") = 1.0, py::arg("plane") = "xy")
      // Deck-driven initial-field seeding. The deck layer supplies only O(1)
      // scalars -- the converted amplitude, the mode numbers, and whatever
      // transcendentals of THOSE the validity refusals already had to compute
      // (omega*dt, the direction ratios, the Bessel root). Every per-lattice
      // value is evaluated on device.
      //
      // Kept as a keyword-argument builder rather than a bound spec class: the
      // caller is the CLI, which assembles the block once and never inspects
      // it, so an attribute-per-field wrapper would be pure ceremony.
      .def("seed_initial_fields",
           [](quasar::pic::EmPic2D3V& self, const std::string& type,
              int component, int magnetic_component, int cylindrical,
              int mode_x, int mode_y, Real amplitude, Real bessel_root,
              Real half_time, Real direction_x, Real direction_y,
              Real magnetic_phase, Real magnetic_sign, Real dt) {
             quasar::pic::PicInitialFieldSpec spec{};
             spec.kind = quasar::pic::pic_initial_field_kind(type);
             spec.cylindrical = cylindrical;
             spec.component = component;
             spec.magnetic_component = magnetic_component;
             spec.mode_x = mode_x;
             spec.mode_y = mode_y;
             spec.amplitude = amplitude;
             spec.bessel_root = bessel_root;
             spec.half_time = half_time;
             spec.direction_x = direction_x;
             spec.direction_y = direction_y;
             spec.magnetic_phase = magnetic_phase;
             spec.magnetic_sign = magnetic_sign;
             self.seed_initial_fields(spec, dt);
           },
           py::arg("type"), py::arg("component"),
           py::arg("magnetic_component") = -1, py::arg("cylindrical") = 0,
           py::arg("mode_x") = 1, py::arg("mode_y") = 0,
           py::arg("amplitude") = 0.0, py::arg("bessel_root") = 0.0,
           py::arg("half_time") = 0.0, py::arg("direction_x") = 0.0,
           py::arg("direction_y") = 0.0, py::arg("magnetic_phase") = 0.0,
           py::arg("magnetic_sign") = 1.0, py::arg("dt") = 0.0)
      .def("seed_field",
           [](quasar::pic::EmPic2D3V& self, const std::string& component,
              const py::object& values) {
             // Writes one Yee field component from a (storage_size,) host array,
             // for deck-driven initial-field seeding. The array must already be in
             // the solver's internal (normalized) units and full ghost-padded
             // storage layout (matches fields_to_host()).
             auto& f = self.fields();
             auto& buf = yee_component_buffer(f, component);
             const auto vec = numpy_to_finite_real_vector(values, "values");
             if (vec.size() != buf.size()) {
               throw std::invalid_argument("seed_field: array size does not match field storage");
             }
             buf.copy_from_host(vec.data(), vec.size());
           },
           py::arg("component"), py::arg("values"))
      .def("storage_size",
           [](quasar::pic::EmPic2D3V& self) { return self.grid().storage_size(); })
      .def("nghost",
           [](const quasar::pic::EmPic2D3V& self) {
             return self.grid().nghost;
           })
      .def("fields_to_host",
           [](quasar::pic::EmPic2D3V& self) {
             return yee_field_to_dict(self.fields());
           })
      .def("field_component_to_host",
           [](quasar::pic::EmPic2D3V& self, const std::string& component) {
             return yee_component_to_numpy(self.fields(), component);
           },
           py::arg("component"))
      .def("external_fields_to_host",
           [](quasar::pic::EmPic2D3V& self) {
             return yee_field_to_dict(self.external_fields());
           })
      .def("external_field_component_to_host",
           [](quasar::pic::EmPic2D3V& self, const std::string& component) {
             return yee_component_to_numpy(self.external_fields(), component);
           },
           py::arg("component"));

  // -- Diagnostics ----------------------------------------------------------

  pic.def("total_kinetic_energy", &quasar::pic::total_kinetic_energy,
          py::arg("species"));
  pic.def("alive_count", &quasar::pic::alive_count, py::arg("species"));
  pic.def("total_em_energy",
          [](quasar::pic::EmPic2D3V& self) {
            return quasar::pic::total_em_energy(self);
          },
          py::arg("solver"),
          "Return the positive same-snapshot Yee field norm. Because E and B "
          "occupy different leapfrog times, this is not the exactly conserved "
          "cross-time Yee invariant.");
  pic.def("gauss_residual",
          [](quasar::pic::EmPic2D3V& self) {
            return quasar::pic::gauss_residual(self);
          },
          py::arg("solver"));
}
