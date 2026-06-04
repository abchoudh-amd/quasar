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

// Forces a contiguous, correctly-typed array at the binding boundary so the raw
// data()+shape(0) copy below never reads strided/wrong-type memory.
using RealArray = py::array_t<Real, py::array::c_style | py::array::forcecast>;

std::vector<Real> numpy_to_real_vector(const RealArray& arr, const char* what) {
  if (arr.ndim() != 1) {
    throw std::invalid_argument(std::string{what} + ": expected 1-D NumPy array");
  }
  const auto* data = arr.data();
  return std::vector<Real>{data, data + arr.shape(0)};
}

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
      .def_static("plasma", &quasar::Normalization::plasma,
                  py::arg("n_ref"), py::arg("q_ref"), py::arg("m_ref"))
      .def("to_internal", &quasar::Normalization::to_internal,
           py::arg("value"), py::arg("tag"))
      .def("to_si", &quasar::Normalization::to_si,
           py::arg("value"), py::arg("tag"))
      .def("length_scale", &quasar::Normalization::length_scale)
      .def("time_scale", &quasar::Normalization::time_scale)
      .def("e_field_scale", &quasar::Normalization::e_field_scale)
      .def("b_field_scale", &quasar::Normalization::b_field_scale)
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
      .def_readwrite("filters", &quasar::pic::EmPicConfig::filters);

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

  // NOTE: ParticleSpecies is move-only (DeviceBuffer is non-copyable). Python
  // users never construct one directly — the solver owns the species and you
  // access it via EmPic2D3V.species_at(idx) after add_species(config).
  py::class_<quasar::pic::ParticleSpecies>(pic, "ParticleSpecies")
      .def_property_readonly("name", &quasar::pic::ParticleSpecies::name)
      .def_property_readonly("charge", &quasar::pic::ParticleSpecies::charge)
      .def_property_readonly("mass", &quasar::pic::ParticleSpecies::mass)
      .def_property_readonly("size", &quasar::pic::ParticleSpecies::size)
      .def_property_readonly("capacity", &quasar::pic::ParticleSpecies::capacity)
      .def("set_host_particles",
           [](quasar::pic::ParticleSpecies& self,
              const RealArray& x, const RealArray& y,
              const RealArray& vx, const RealArray& vy,
              const RealArray& vz, const RealArray& w) {
             self.set_host_particles(
                 numpy_to_real_vector(x, "x"), numpy_to_real_vector(y, "y"),
                 numpy_to_real_vector(vx, "vx"), numpy_to_real_vector(vy, "vy"),
                 numpy_to_real_vector(vz, "vz"), numpy_to_real_vector(w, "weight"));
           },
           py::arg("x"), py::arg("y"),
           py::arg("vx"), py::arg("vy"), py::arg("vz"),
           py::arg("weight"))
      .def("to_host",
           [](const quasar::pic::ParticleSpecies& self) {
             auto snap = self.to_host();
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
           [](quasar::pic::EmPic2D3V& self, std::size_t idx)
               -> quasar::pic::ParticleSpecies& {
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
                                                b_field_scale, plane);
           },
           py::arg("evaluator"), py::arg("conductors"),
           py::arg("length_scale") = 1.0, py::arg("e_field_scale") = 1.0,
           py::arg("b_field_scale") = 1.0, py::arg("plane") = "xy")
      .def("seed_field",
           [](quasar::pic::EmPic2D3V& self, const std::string& component,
              const RealArray& values) {
             // Writes one Yee field component from a (storage_size,) host array,
             // for deck-driven initial-field seeding. The array must already be in
             // the solver's internal (normalized) units and full ghost-padded
             // storage layout (matches fields_to_host()).
             auto& f = self.fields();
             auto& buf = yee_component_buffer(f, component);
             const auto vec = numpy_to_real_vector(values, "values");
             if (vec.size() != buf.size()) {
               throw std::invalid_argument("seed_field: array size does not match field storage");
             }
             buf.copy_from_host(vec.data(), vec.size());
           },
           py::arg("component"), py::arg("values"))
      .def("storage_size",
           [](quasar::pic::EmPic2D3V& self) { return self.grid().storage_size(); })
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
            return quasar::pic::total_em_energy(self.fields(), self.grid());
          },
          py::arg("solver"));
  pic.def("gauss_residual",
          [](quasar::pic::EmPic2D3V& self) {
            return quasar::pic::gauss_residual(self.fields(), self.current());
          },
          py::arg("solver"));
}
