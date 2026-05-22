#include "quasar/numerics/filter.hpp"

#include <gtest/gtest.h>

TEST(PicFilterChargeConservation, PipelineCanOwnFilters) {
  quasar::numerics::FilterPipeline pipeline;
  pipeline.add(std::make_unique<quasar::numerics::BinomialFilter>(1));
  EXPECT_EQ(pipeline.size(), 1u);
}
