#pragma once

#include "quasar/distributed/mpi_runtime.hpp"

#include <mpi.h>

namespace quasar::distributed::detail {

// Private bridge for translation units that must pass a native communicator
// to MPI or parallel-HDF5.  Keeping this bridge under src/ prevents MPI types
// and headers from leaking through Quasar's installed public include tree.
class MpiRuntimeNativeAccess {
 public:
  [[nodiscard]] static MPI_Comm world(const MpiRuntime& runtime);
  [[nodiscard]] static MPI_Comm shared_memory(const MpiRuntime& runtime);
};

}  // namespace quasar::distributed::detail
