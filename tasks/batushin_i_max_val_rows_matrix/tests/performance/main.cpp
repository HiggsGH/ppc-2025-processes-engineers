#include <gtest/gtest.h>

#include "batushin_i_max_val_rows_matrix/common/include/common.hpp"
#include "batushin_i_max_val_rows_matrix/mpi/include/ops_mpi.hpp"
#include "batushin_i_max_val_rows_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace batushin_i_max_val_rows_matrix {

class BatushinIMaxValRowsMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(BatushinIMaxValRowsMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BatushinIMaxValRowsMatrixMPI, BatushinIMaxValRowsMatrixSEQ>(PPC_SETTINGS_batushin_i_max_val_rows_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BatushinIMaxValRowsMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BatushinIMaxValRowsMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace batushin_i_max_val_rows_matrix
