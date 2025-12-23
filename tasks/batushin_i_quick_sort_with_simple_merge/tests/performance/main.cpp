#include <gtest/gtest.h>

#include <random>

#include "batushin_i_quick_sort_with_simple_merge/common/include/common.hpp"
#include "batushin_i_quick_sort_with_simple_merge/mpi/include/ops_mpi.hpp"
#include "batushin_i_quick_sort_with_simple_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace batushin_i_quick_sort_with_simple_merge {

class BatushinIQuickSortWithSimpleMergePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  OutType expected_result_;

 public:
  void SetUp() override {
    const size_t N = 5000000;
    std::vector<int> data(N);

    unsigned int seed = 123456789;
    for (size_t i = 0; i < N; ++i) {
      seed = seed * 1103515245 + 12345;
      data[i] = static_cast<int>(seed % 2000001) - 1000000;
    }

    input_data_ = data;
    expected_result_ = data;
    std::sort(expected_result_.begin(), expected_result_.end());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_result_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BatushinIQuickSortWithSimpleMergePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BatushinIQuickSortWithSimpleMergeMPI, BatushinIQuickSortWithSimpleMergeSEQ>(
        PPC_SETTINGS_batushin_i_quick_sort_with_simple_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BatushinIQuickSortWithSimpleMergePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BatushinIQuickSortWithSimpleMergePerfTests, kGtestValues, kPerfTestName);

}  // namespace batushin_i_quick_sort_with_simple_merge
