#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "batushin_i_max_val_rows_matrix/common/include/common.hpp"
#include "batushin_i_max_val_rows_matrix/mpi/include/ops_mpi.hpp"
#include "batushin_i_max_val_rows_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace batushin_i_max_val_rows_matrix {

class BatushinIMaxValRowsMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  OutType expected_result_;

 public:
  void SetUp() override {
    const size_t rows = 5000;
    const size_t columns = 5000;

    std::vector<double> matrix_data(rows * columns);

    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < columns; j++) {
        matrix_data[(i * columns) + j] = static_cast<double>(j);
      }
    }

    input_data_ = std::make_tuple(rows, columns, matrix_data);

    expected_result_ = std::vector<double>(rows, static_cast<double>(columns - 1));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_result_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BatushinIMaxValRowsMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BatushinIMaxValRowsMatrixMPI, BatushinIMaxValRowsMatrixSEQ>(
        PPC_SETTINGS_batushin_i_max_val_rows_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BatushinIMaxValRowsMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BatushinIMaxValRowsMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace batushin_i_max_val_rows_matrix
