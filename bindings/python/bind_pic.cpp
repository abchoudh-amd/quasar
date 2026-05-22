#include <pybind11/pybind11.h>

#include "quasar/core/grid.hpp"
#include "quasar/core/normalization.hpp"
#include "quasar/physics/pic/pic_solver.hpp"

namespace py = pybind11;

void bind_pic(py::module_& m) {
  py::module_ pic = m.def_submodule("pic", "2D3V electromagnetic PIC scaffolding.");

  py::class_<quasar::Grid2D>(pic, "Grid2D")
      .def(py::init<int, int, quasar::Real, quasar::Real,
                    quasar::Real, quasar::Real, int>(),
           py::arg("nx"), py::arg("ny"), py::arg("lx"), py::arg("ly"),
           py::arg("origin_x") = 0.0, py::arg("origin_y") = 0.0,
           py::arg("nghost") = 1)
      .def_readwrite("nx", &quasar::Grid2D::nx)
      .def_readwrite("ny", &quasar::Grid2D::ny)
      .def_readwrite("lx", &quasar::Grid2D::lx)
      .def_readwrite("ly", &quasar::Grid2D::ly)
      .def("dx", &quasar::Grid2D::dx)
      .def("dy", &quasar::Grid2D::dy);

  py::class_<quasar::Normalization>(pic, "Normalization")
      .def_static("plasma", &quasar::Normalization::plasma)
      .def_readonly("omega_p_ref", &quasar::Normalization::omega_p_ref);

  py::class_<quasar::pic::EmPicConfig>(pic, "EmPicConfig")
      .def(py::init<>())
      .def_readwrite("grid", &quasar::pic::EmPicConfig::grid)
      .def_readwrite("fdtd_order", &quasar::pic::EmPicConfig::fdtd_order)
      .def_readwrite("shape_order", &quasar::pic::EmPicConfig::shape_order);

  py::class_<quasar::pic::EmPic2D3V>(pic, "EmPic2D3V")
      .def(py::init<quasar::pic::EmPicConfig>())
      .def("step", &quasar::pic::EmPic2D3V::step)
      .def("advance", &quasar::pic::EmPic2D3V::advance);
}
