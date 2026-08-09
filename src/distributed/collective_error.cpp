#include "quasar/distributed/collective_error.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <tuple>

namespace quasar::distributed {
namespace {

template <std::size_t N>
void copy_text(std::array<char, N>& destination,
               std::string_view source) noexcept {
  static_assert(N > 0);
  const std::size_t count = std::min(source.size(), N - 1);
  if (count != 0) std::memcpy(destination.data(), source.data(), count);
  destination[count] = '\0';
}

template <std::size_t N>
std::string_view read_text(const std::array<char, N>& source) noexcept {
  const auto end = std::find(source.begin(), source.end(), '\0');
  return {source.data(), static_cast<std::size_t>(end - source.begin())};
}

CollectiveErrorRecord make_record(ErrorDisposition disposition,
                                  std::uint64_t epoch,
                                  std::int32_t rank,
                                  std::int32_t endpoint,
                                  std::int32_t code,
                                  std::string_view phase,
                                  std::string_view message) noexcept {
  CollectiveErrorRecord result;
  result.epoch       = epoch;
  result.rank        = rank;
  result.endpoint    = endpoint;
  result.code        = code;
  result.disposition = disposition;
  copy_text(result.phase, phase);
  copy_text(result.message, message);
  return result;
}

CollectiveResolution protocol_failure(
    std::span<const CollectiveErrorRecord> records,
    std::string_view message) noexcept {
  const std::uint64_t epoch = records.empty() ? 0 : records.front().epoch;
  CollectiveResolution result;
  result.decision          = CollectiveDecision::fail;
  result.representative    = make_record(
      ErrorDisposition::fatal, epoch, -1, -1, collective_protocol_error,
      "collective-consensus", message);
  result.participant_count = records.size();
  result.failure_count     = 1;
  return result;
}

bool valid_disposition(ErrorDisposition disposition) noexcept {
  return disposition == ErrorDisposition::success
      || disposition == ErrorDisposition::retry
      || disposition == ErrorDisposition::fatal;
}

}  // namespace

CollectiveErrorRecord CollectiveErrorRecord::success(
    std::uint64_t epoch, std::int32_t rank, std::string_view phase) {
  return make_record(ErrorDisposition::success, epoch, rank, -1, 0,
                     phase, {});
}

CollectiveErrorRecord CollectiveErrorRecord::retry(
    std::uint64_t epoch, std::int32_t rank, std::int32_t endpoint,
    std::int32_t code, std::string_view phase, std::string_view message) {
  return make_record(ErrorDisposition::retry, epoch, rank, endpoint, code,
                     phase, message);
}

CollectiveErrorRecord CollectiveErrorRecord::failure(
    std::uint64_t epoch, std::int32_t rank, std::int32_t endpoint,
    std::int32_t code, std::string_view phase, std::string_view message) {
  return make_record(ErrorDisposition::fatal, epoch, rank, endpoint, code,
                     phase, message);
}

std::string_view CollectiveErrorRecord::phase_text() const noexcept {
  return read_text(phase);
}

std::string_view CollectiveErrorRecord::message_text() const noexcept {
  return read_text(message);
}

CollectiveResolution resolve_collective_errors(
    std::span<const CollectiveErrorRecord> records) noexcept {
  if (records.empty()) {
    return protocol_failure(records,
                            "collective status set has no participants");
  }

  // Validate the wire values before using them to check coverage.  Keeping
  // this pass first also makes an out-of-range rank distinguishable from an
  // otherwise incomplete rank set.
  for (const auto& record : records) {
    if (record.rank < 0
        || static_cast<std::size_t>(record.rank) >= records.size()) {
      return protocol_failure(records,
                              "collective status contains an out-of-range rank");
    }
    if (!valid_disposition(record.disposition)) {
      return protocol_failure(records,
                              "collective status contains an invalid disposition");
    }
  }

  // Check rank coverage without allocating: every rank in [0,N) must occur
  // exactly once.
  for (std::size_t expected = 0; expected < records.size(); ++expected) {
    std::size_t occurrences = 0;
    for (const auto& record : records) {
      if (record.rank == static_cast<std::int32_t>(expected)) ++occurrences;
    }
    if (occurrences != 1) {
      return protocol_failure(records,
                              "collective status ranks are incomplete or duplicated");
    }
  }
  const std::uint64_t epoch = records.front().epoch;
  const std::string_view phase = records.front().phase_text();
  for (const auto& record : records) {
    if (record.epoch != epoch) {
      return protocol_failure(records,
                              "collective records disagree on communication epoch");
    }
    if (record.phase_text() != phase) {
      return protocol_failure(records,
                              "collective records disagree on execution phase");
    }
  }

  CollectiveResolution result;
  result.participant_count = records.size();
  for (const auto& record : records) {
    if (record.disposition == ErrorDisposition::retry) ++result.retry_count;
    if (record.disposition == ErrorDisposition::fatal) ++result.failure_count;
  }

  ErrorDisposition target = ErrorDisposition::success;
  if (result.failure_count != 0) {
    target          = ErrorDisposition::fatal;
    result.decision = CollectiveDecision::fail;
  } else if (result.retry_count != 0) {
    target          = ErrorDisposition::retry;
    result.decision = CollectiveDecision::retry;
  }

  // Select the lowest-rank (then endpoint/code) record at the winning
  // severity.  This makes the exception text independent of allgather order.
  const CollectiveErrorRecord* representative = nullptr;
  for (const auto& record : records) {
    if (record.disposition != target) continue;
    const auto key = [](const CollectiveErrorRecord& value) {
      return std::tuple{value.rank, value.endpoint, value.code};
    };
    if (representative == nullptr || key(record) < key(*representative)) {
      representative = &record;
    }
  }
  result.representative = *representative;
  return result;
}

}  // namespace quasar::distributed
