#include <gtest/gtest.h>

#include "batushin_i_striped_matrix_multiplication/common/include/common.hpp"
#include "batushin_i_striped_matrix_multiplication/mpi/include/ops_mpi.hpp"
#include "batushin_i_striped_matrix_multiplication/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace batushin_i_striped_matrix_multiplication {

class BatushinIStripedMatrixMultiplicationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BatushinIStripedMatrixMultiplicationPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BatushinIStripedMatrixMultiplicationMPI, BatushinIStripedMatrixMultiplicationSEQ>(PPC_SETTINGS_batushin_i_striped_matrix_multiplication);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BatushinIStripedMatrixMultiplicationPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BatushinIStripedMatrixMultiplicationPerfTests, kGtestValues, kPerfTestName);

}  // namespace batushin_i_striped_matrix_multiplication
