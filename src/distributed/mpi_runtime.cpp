#include "mpi_runtime_native.hpp"

#include "fixed_message.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace quasar::distributed {

class MpiRuntime::Impl {
 public:
  std::thread::id orchestration_thread{std::this_thread::get_id()};
  MPI_Comm world{MPI_COMM_NULL};
  MPI_Comm shared_memory{MPI_COMM_NULL};
  int rank{-1};
  int size{0};
  int node_rank{-1};
  int node_size{0};
  int thread_level{MPI_THREAD_SINGLE};
  bool owns_mpi{false};
  bool closed{false};
  // Allocated once after communicator sizing and reused only on the
  // orchestration thread when a consensus failure needs detailed records.
  std::unique_ptr<CollectiveErrorRecord[]> consensus_records{};
};

namespace {

void write_collective_error_message(
    const CollectiveResolution& resolution,
    std::span<char> storage) noexcept {
  const auto& record = resolution.representative;
  FixedMessageWriter message{storage};
  message.append("distributed collective ");
  message.append(resolution.decision == CollectiveDecision::retry
                     ? "requested retry"
                     : "failed");
  message.append(" during '");
  message.append(record.phase_text());
  message.append("' at epoch ");
  message.append_integer(record.epoch);
  message.append(" on rank ");
  message.append_integer(record.rank);
  if (record.endpoint >= 0) {
    message.append(", endpoint ");
    message.append_integer(record.endpoint);
  }
  if (record.code != 0) {
    message.append(" (code ");
    message.append_integer(record.code);
    message.append(")");
  }
  if (!record.message_text().empty()) {
    message.append(": ");
    message.append(record.message_text());
  }
}

template <class T>
T allreduce_scalar(MPI_Comm communicator, T local, MPI_Datatype datatype,
                   MPI_Op operation, const char* name) {
  T result{};
  check_mpi(MPI_Allreduce(&local, &result, 1, datatype, operation,
                          communicator),
            name);
  return result;
}

bool construction_all(MPI_Comm communicator, bool local,
                      const char* operation) {
  const int value = local ? 1 : 0;
  int result = 0;
  check_mpi(MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MIN,
                          communicator),
            operation);
  return result != 0;
}

// Summarize the success-path protocol in one small reduction.  For every
// value, reduce both its bits and their complement with bitwise OR.  The
// reduced pair is equal/complementary iff every participant supplied exactly
// the same value.  This preserves epoch/phase validation without gathering a
// full CollectiveErrorRecord from every rank on the common success path.
constexpr std::size_t consensus_phase_words =
    (collective_phase_capacity + sizeof(std::uint64_t) - 1)
    / sizeof(std::uint64_t);
constexpr std::size_t consensus_value_words = 2 + consensus_phase_words;
constexpr std::size_t consensus_summary_words = 2 * consensus_value_words;

using ConsensusSummary =
    std::array<std::uint64_t, consensus_summary_words>;

ConsensusSummary make_consensus_summary(
    const CollectiveErrorRecord& local) noexcept {
  std::array<std::uint64_t, consensus_value_words> values{};
  values[0] = local.ok() ? 0U : 1U;
  values[1] = local.epoch;

  std::array<char, collective_phase_capacity> canonical_phase{};
  const std::string_view phase = local.phase_text();
  if (!phase.empty()) {
    std::memcpy(canonical_phase.data(), phase.data(), phase.size());
  }
  std::memcpy(values.data() + 2, canonical_phase.data(),
              canonical_phase.size());

  ConsensusSummary result{};
  for (std::size_t i = 0; i < values.size(); ++i) {
    result[2 * i] = values[i];
    result[2 * i + 1] = ~values[i];
  }
  return result;
}

bool consensus_values_agree(const ConsensusSummary& summary) noexcept {
  // Word zero is the any-rank-needs-detail flag.  Once it is set, the full
  // records are required even if every rank reported the same failure kind.
  if (summary[0] != 0U) return false;
  for (std::size_t i = 1; i < consensus_value_words; ++i) {
    if (summary[2 * i] != ~summary[2 * i + 1]) return false;
  }
  return true;
}

}  // namespace

std::string mpi_error_string(int code) {
  std::array<char, MPI_MAX_ERROR_STRING> storage{};
  int length = 0;
  const int status = MPI_Error_string(code, storage.data(), &length);
  if (status != MPI_SUCCESS || length <= 0
      || length > MPI_MAX_ERROR_STRING) {
    return "MPI error " + std::to_string(code);
  }
  return std::string{storage.data(), static_cast<std::size_t>(length)};
}

void check_mpi(int code, const char* operation) {
  if (code == MPI_SUCCESS) return;
  throw MpiError{operation == nullptr ? "MPI operation" : operation,
                 code, mpi_error_string(code)};
}

MpiError::MpiError(std::string operation, int code, std::string detail)
  : std::runtime_error{operation + " failed: " + detail},
    operation_{std::move(operation)},
    code_{code} {}

DistributedCollectiveError::DistributedCollectiveError(
    CollectiveResolution resolution) noexcept
  : resolution_{std::move(resolution)} {
  write_collective_error_message(resolution_, message_);
}

MpiRuntime::MpiRuntime(int* argc, char*** argv)
  : impl_{std::make_unique<Impl>()} {
  try {
    int finalized = 0;
    check_mpi(MPI_Finalized(&finalized), "MPI_Finalized");
    if (finalized != 0) {
      throw std::runtime_error{
          "distributed runtime cannot start after MPI has been finalized"};
    }

    int initialized = 0;
    check_mpi(MPI_Initialized(&initialized), "MPI_Initialized");
    if (initialized == 0) {
      int local_argc = 0;
      char** local_argv = nullptr;
      int* init_argc = argc == nullptr ? &local_argc : argc;
      char*** init_argv = argv == nullptr ? &local_argv : argv;
      const int status = MPI_Init_thread(
          init_argc, init_argv, MPI_THREAD_FUNNELED, &impl_->thread_level);
      if (status != MPI_SUCCESS) {
        throw MpiError{"MPI_Init_thread", status,
                       "MPI_Init_thread returned status "
                           + std::to_string(status)};
      }
      impl_->owns_mpi = true;
    } else {
      check_mpi(MPI_Query_thread(&impl_->thread_level), "MPI_Query_thread");
    }

    // Thread support is a rank-local property, but construction is collective.
    // Reach one agreement point before any communicator duplication so a bad
    // participant cannot throw while its peers enter MPI_Comm_dup/split.
    int is_main = 0;
    const int main_status = MPI_Is_thread_main(&is_main);
    const bool local_thread_valid =
        main_status == MPI_SUCCESS
        && impl_->thread_level >= MPI_THREAD_FUNNELED
        && (impl_->thread_level != MPI_THREAD_FUNNELED || is_main != 0);
    const int local_ok = local_thread_valid ? 1 : 0;
    int all_ok = 0;
    check_mpi(MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN,
                            MPI_COMM_WORLD),
              "MPI_Allreduce(construction thread support)");
    if (all_ok == 0) {
      throw std::runtime_error{
          "at least one MPI rank does not provide usable MPI_THREAD_FUNNELED "
          "support on the orchestration thread"};
    }

    int status = MPI_Comm_dup(MPI_COMM_WORLD, &impl_->world);
    if (!construction_all(
            MPI_COMM_WORLD,
            status == MPI_SUCCESS && impl_->world != MPI_COMM_NULL,
            "MPI_Allreduce(construction world duplication)")) {
      // A communicator may exist on only a subset after a failed collective.
      // It cannot be safely freed collectively, so abandon the local handle
      // and make every rank leave construction at the same phase.
      impl_->world = MPI_COMM_NULL;
      throw std::runtime_error{
          "at least one MPI rank could not duplicate the world communicator"};
    }

    status = MPI_Comm_set_errhandler(impl_->world, MPI_ERRORS_RETURN);
    if (!construction_all(
            impl_->world, status == MPI_SUCCESS,
            "MPI_Allreduce(construction world error handler)")) {
      throw std::runtime_error{
          "at least one MPI rank could not set the world error handler"};
    }

    const int rank_status = MPI_Comm_rank(impl_->world, &impl_->rank);
    const int size_status = MPI_Comm_size(impl_->world, &impl_->size);
    if (!construction_all(
            impl_->world,
            rank_status == MPI_SUCCESS && size_status == MPI_SUCCESS,
            "MPI_Allreduce(construction world metadata)")) {
      throw std::runtime_error{
          "at least one MPI rank could not query world communicator metadata"};
    }

    impl_->consensus_records.reset(new (std::nothrow) CollectiveErrorRecord[
        static_cast<std::size_t>(impl_->size)]);
    if (!construction_all(
            impl_->world, impl_->consensus_records != nullptr,
            "MPI_Allreduce(construction consensus storage)")) {
      throw std::runtime_error{
          "at least one MPI rank could not allocate collective status storage"};
    }

    status = MPI_Comm_split_type(impl_->world, MPI_COMM_TYPE_SHARED,
                                 impl_->rank, MPI_INFO_NULL,
                                 &impl_->shared_memory);
    if (!construction_all(
            impl_->world, status == MPI_SUCCESS
                              && impl_->shared_memory != MPI_COMM_NULL,
            "MPI_Allreduce(construction shared communicator)")) {
      impl_->shared_memory = MPI_COMM_NULL;
      throw std::runtime_error{
          "at least one MPI rank could not create the shared-memory communicator"};
    }

    const int shared_handler_status =
        MPI_Comm_set_errhandler(impl_->shared_memory, MPI_ERRORS_RETURN);
    const int node_rank_status =
        MPI_Comm_rank(impl_->shared_memory, &impl_->node_rank);
    const int node_size_status =
        MPI_Comm_size(impl_->shared_memory, &impl_->node_size);
    if (!construction_all(
            impl_->world, shared_handler_status == MPI_SUCCESS
                              && node_rank_status == MPI_SUCCESS
                              && node_size_status == MPI_SUCCESS,
            "MPI_Allreduce(construction shared metadata)")) {
      throw std::runtime_error{
          "at least one MPI rank could not configure shared-memory metadata"};
    }
  } catch (...) {
    // A throwing constructor has no destructor.  Release every successfully
    // created handle and undo only the MPI lifetime this object started.  No
    // barrier is attempted: construction may have failed before the private
    // world communicator became usable.
    if (impl_->shared_memory != MPI_COMM_NULL) {
      (void)MPI_Comm_free(&impl_->shared_memory);
    }
    if (impl_->world != MPI_COMM_NULL) (void)MPI_Comm_free(&impl_->world);
    if (impl_->owns_mpi) {
      (void)MPI_Finalize();
      impl_->owns_mpi = false;
    }
    impl_->closed = true;
    throw;
  }
}

MpiRuntime::~MpiRuntime() noexcept = default;

int MpiRuntime::rank() const noexcept { return impl_->rank; }

int MpiRuntime::size() const noexcept { return impl_->size; }

int MpiRuntime::node_rank() const noexcept { return impl_->node_rank; }

int MpiRuntime::node_size() const noexcept { return impl_->node_size; }

int MpiRuntime::thread_level() const noexcept { return impl_->thread_level; }

bool MpiRuntime::owns_mpi() const noexcept { return impl_->owns_mpi; }

bool MpiRuntime::closed() const noexcept { return impl_->closed; }

void MpiRuntime::require_orchestration_thread() const {
  if (std::this_thread::get_id() != impl_->orchestration_thread) {
    throw std::logic_error{
        "all MPI and parallel-HDF5 calls must run on the orchestration thread"};
  }
}

void MpiRuntime::require_open() const {
  require_orchestration_thread();
  if (impl_->closed || impl_->world == MPI_COMM_NULL) {
    throw std::logic_error{"distributed runtime is closed"};
  }
}

MPI_Comm detail::MpiRuntimeNativeAccess::world(const MpiRuntime& runtime) {
  runtime.require_open();
  return runtime.impl_->world;
}

MPI_Comm detail::MpiRuntimeNativeAccess::shared_memory(
    const MpiRuntime& runtime) {
  runtime.require_open();
  return runtime.impl_->shared_memory;
}

void MpiRuntime::barrier() const {
  require_open();
  check_mpi(MPI_Barrier(impl_->world), "MPI_Barrier");
}

CollectiveResolution MpiRuntime::consensus(
    CollectiveErrorRecord local) const {
  require_open();
  local.rank = impl_->rank;
  if (sizeof(CollectiveErrorRecord)
      > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error{"collective error record exceeds MPI count range"};
  }
  // Successful worker phases dominate normal execution.  One compact
  // reduction validates both the status and the exact epoch/phase signature;
  // only a retry, failure, or protocol mismatch gathers full records.  The
  // receive storage was allocated collectively once during construction, so
  // consensus performs no heap work.
  const ConsensusSummary local_summary = make_consensus_summary(local);
  ConsensusSummary global_summary{};
  check_mpi(MPI_Allreduce(local_summary.data(), global_summary.data(),
                          static_cast<int>(global_summary.size()),
                          MPI_UINT64_T, MPI_BOR, impl_->world),
            "MPI_Allreduce(collective status summary)");
  if (consensus_values_agree(global_summary)) {
    CollectiveResolution result;
    result.decision = CollectiveDecision::accept;
    result.representative = CollectiveErrorRecord::success(
        local.epoch, 0, local.phase_text());
    result.participant_count = static_cast<std::size_t>(impl_->size);
    return result;
  }
  check_mpi(MPI_Allgather(
                &local, static_cast<int>(sizeof(local)), MPI_BYTE,
                impl_->consensus_records.get(),
                static_cast<int>(sizeof(local)), MPI_BYTE, impl_->world),
            "MPI_Allgather(collective status)");
  return resolve_collective_errors(
      std::span<const CollectiveErrorRecord>{
          impl_->consensus_records.get(),
          static_cast<std::size_t>(impl_->size)});
}

void MpiRuntime::require_collective_success(
    CollectiveErrorRecord local) const {
  const CollectiveResolution resolution = consensus(local);
  if (!resolution.accepted()) {
    throw DistributedCollectiveError{resolution};
  }
}

double MpiRuntime::allreduce_max(double local) const {
  require_open();
  return allreduce_scalar(impl_->world, local, MPI_DOUBLE, MPI_MAX,
                          "MPI_Allreduce(max double)");
}

double MpiRuntime::allreduce_min(double local) const {
  require_open();
  return allreduce_scalar(impl_->world, local, MPI_DOUBLE, MPI_MIN,
                          "MPI_Allreduce(min double)");
}

double MpiRuntime::allreduce_sum(double local) const {
  require_open();
  return allreduce_scalar(impl_->world, local, MPI_DOUBLE, MPI_SUM,
                          "MPI_Allreduce(sum double)");
}

std::uint64_t MpiRuntime::allreduce_sum(std::uint64_t local) const {
  require_open();
  return allreduce_scalar(impl_->world, local, MPI_UINT64_T, MPI_SUM,
                          "MPI_Allreduce(sum uint64)");
}

bool MpiRuntime::allreduce_all(bool local) const {
  require_open();
  const int value = local ? 1 : 0;
  return allreduce_scalar(impl_->world, value, MPI_INT, MPI_MIN,
                          "MPI_Allreduce(all bool)") != 0;
}

void MpiRuntime::close() {
  require_orchestration_thread();
  if (impl_->closed) return;

  int first_error = MPI_SUCCESS;
  const auto remember = [&first_error](int status) noexcept {
    if (first_error == MPI_SUCCESS && status != MPI_SUCCESS) {
      first_error = status;
    }
  };

  if (impl_->world != MPI_COMM_NULL) remember(MPI_Barrier(impl_->world));
  if (impl_->shared_memory != MPI_COMM_NULL) {
    remember(MPI_Comm_free(&impl_->shared_memory));
  }
  if (impl_->world != MPI_COMM_NULL) remember(MPI_Comm_free(&impl_->world));

  // Formatting an MPI error into std::string can allocate.  Do not do that
  // between communicator collectives.  Capture only into fixed stack storage
  // once every communicator has been released and while MPI is still live.
  std::array<char, MPI_MAX_ERROR_STRING> error_storage{};
  int error_length = 0;
  if (first_error != MPI_SUCCESS) {
    const int text_status =
        MPI_Error_string(first_error, error_storage.data(), &error_length);
    if (text_status != MPI_SUCCESS || error_length < 0
        || error_length > MPI_MAX_ERROR_STRING) {
      error_length = 0;
    }
  }

  impl_->closed = true;
  if (impl_->owns_mpi) {
    const int finalize_status = MPI_Finalize();
    if (first_error == MPI_SUCCESS && finalize_status != MPI_SUCCESS) {
      first_error = finalize_status;
      error_length = 0;
    }
    impl_->owns_mpi = false;
  }
  if (first_error != MPI_SUCCESS) {
    std::string first_error_detail;
    if (error_length > 0) {
      first_error_detail.assign(
          error_storage.data(), static_cast<std::size_t>(error_length));
    } else {
      first_error_detail = "MPI close returned status "
                         + std::to_string(first_error);
    }
    throw MpiError{"distributed runtime close", first_error,
                   std::move(first_error_detail)};
  }
}

}  // namespace quasar::distributed
