#pragma once

#include <pybind11/numpy.h>

#include "quasar/core/types.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace quasar::python_detail {

namespace py = pybind11;

// Preserve the convenient binding contract (array-like input, arbitrary real
// integer/float precision, and non-contiguous views) without NumPy's unsafe
// complex-to-real and bool-to-number coercions.  Inspect the original dtype
// before forcecasting: after py::array_t<Real, forcecast> conversion the lost
// imaginary part or boolean provenance cannot be recovered.
using RealArray = py::array_t<Real, py::array::c_style | py::array::forcecast>;

inline RealArray require_real_array(py::handle value, const char* what) {
  const py::array input = py::array::ensure(value);
  if (!input) {
    throw std::invalid_argument{
        std::string{what} + ": expected a NumPy-compatible array"};
  }

  const char kind = input.dtype().kind();
  if (kind != 'f' && kind != 'i' && kind != 'u') {
    throw std::invalid_argument{
        std::string{what} +
        ": expected a real floating-point or integer array (bool, complex, "
        "object, string, void, and datetime dtypes are not accepted)"};
  }

  RealArray converted = RealArray::ensure(input);
  if (!converted) {
    throw std::invalid_argument{
        std::string{what} + ": values cannot be represented as float64"};
  }
  return converted;
}

inline std::vector<Real> numpy_to_real_vector(py::handle value,
                                               const char* what) {
  const RealArray array = require_real_array(value, what);
  if (array.ndim() != 1) {
    throw std::invalid_argument{
        std::string{what} + ": expected 1-D NumPy array"};
  }
  const auto* data = array.data();
  return std::vector<Real>{data, data + array.shape(0)};
}

inline std::vector<Real> numpy_to_finite_real_vector(py::handle value,
                                                      const char* what) {
  auto result = numpy_to_real_vector(value, what);
  if (!std::all_of(result.begin(), result.end(),
                   [](Real item) { return std::isfinite(item); })) {
    throw std::invalid_argument{
        std::string{what} + ": expected only finite values"};
  }
  return result;
}

}  // namespace quasar::python_detail
