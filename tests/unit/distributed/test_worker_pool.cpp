#include "quasar/backend/device.hpp"
#include "quasar/distributed/worker_pool.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace {

constexpr int kSkip = 77;

}  // namespace

TEST(DistributedWorkerPool, PersistentWorkersOwnDistinctDevices) {
  if (quasar::backend::device_count() < 2) {
    GTEST_SKIP() << "requires two visible GPUs";
  }
  const std::vector<int> devices{0, 1};
  quasar::distributed::EndpointWorkerPool workers{devices, 4};
  std::vector<int> observed_device(2, -1);
  std::vector<std::thread::id> observed_thread(2);
  std::vector<quasar::distributed::WorkerTask> tasks;
  for (std::size_t index = 0; index < devices.size(); ++index) {
    tasks.emplace_back([&, index](auto& context) {
      observed_device[index] = quasar::backend::current_device();
      observed_thread[index] = std::this_thread::get_id();
      EXPECT_EQ(context.endpoint, 4 + index);
      EXPECT_EQ(context.rank_local_index, index);
      EXPECT_EQ(context.device_ordinal, devices[index]);
    });
  }

  const auto status = workers.execute(3, 0, "worker-ownership", tasks);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(observed_device, devices);
  EXPECT_NE(observed_thread[0], observed_thread[1]);
  workers.close();
}

TEST(DistributedWorkerPool, WaitsForEveryEndpointBeforeReportingRetry) {
  if (quasar::backend::device_count() < 2) {
    GTEST_SKIP() << "requires two visible GPUs";
  }
  const std::vector<int> devices{0, 1};
  quasar::distributed::EndpointWorkerPool workers{devices};
  std::atomic<int> completed{0};
  std::vector<quasar::distributed::WorkerTask> tasks;
  tasks.emplace_back([&](auto&) {
    ++completed;
    throw quasar::distributed::RetryableWorkerError{27, "retry"};
  });
  tasks.emplace_back([&](auto&) { ++completed; });

  const auto status = workers.execute(8, 2, "positivity", tasks);
  EXPECT_EQ(completed.load(), 2);
  EXPECT_EQ(status.disposition,
            quasar::distributed::ErrorDisposition::retry);
  EXPECT_EQ(status.rank, 2);
  EXPECT_EQ(status.endpoint, 0);
  EXPECT_EQ(status.code, 27);
  workers.close();
}

TEST(DistributedWorkerPool, FatalEndpointFailureTakesPrecedenceOverRetry) {
  if (quasar::backend::device_count() < 2) {
    GTEST_SKIP() << "requires two visible GPUs";
  }
  const std::vector<int> devices{0, 1};
  quasar::distributed::EndpointWorkerPool workers{devices};
  std::vector<quasar::distributed::WorkerTask> tasks;
  tasks.emplace_back([](auto&) {
    throw quasar::distributed::RetryableWorkerError{27, "retry"};
  });
  tasks.emplace_back([](auto&) { throw std::runtime_error{"fatal"}; });

  const auto status = workers.execute(9, 2, "phase", tasks);
  EXPECT_EQ(status.disposition,
            quasar::distributed::ErrorDisposition::fatal);
  EXPECT_EQ(status.endpoint, 1);
  EXPECT_EQ(status.message_text(), "fatal");
  workers.close();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (quasar::backend::device_count() < 2) {
    return kSkip;
  }
  return RUN_ALL_TESTS();
}
