#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "batushin_i_striped_matrix_multiplication/common/include/common.hpp"
#include "batushin_i_striped_matrix_multiplication/mpi/include/ops_mpi.hpp"
#include "batushin_i_striped_matrix_multiplication/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace batushin_i_striped_matrix_multiplication {

class BatushinIStripedMatrixMultiplicationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;

 public:
  void SetUp() override {
    const size_t rows_a = 1000;
    const size_t columns_a = 1000;
    const size_t rows_b = 1000;
    const size_t columns_b = 1000;

    std::vector<double> matrix_a(rows_a * columns_a);
    for (size_t i = 0; i < rows_a; i++) {
      for (size_t j = 0; j < columns_a; j++) {
        matrix_a[(i * columns_a) + j] = static_cast<double>(i + j);
      }
    }

    std::vector<double> matrix_b(rows_b * columns_b);
    for (size_t i = 0; i < rows_b; i++) {
      for (size_t j = 0; j < columns_b; j++) {
        matrix_b[(i * columns_b) + j] = static_cast<double>(i * j);
      }
    }

    input_data_ = std::make_tuple(rows_a, columns_a, matrix_a, rows_b, columns_b, matrix_b);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BatushinIStripedMatrixMultiplicationPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, BatushinIStripedMatrixMultiplicationMPI,
                                                       BatushinIStripedMatrixMultiplicationSEQ>(
    PPC_SETTINGS_batushin_i_striped_matrix_multiplication);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BatushinIStripedMatrixMultiplicationPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BatushinIStripedMatrixMultiplicationPerfTests, kGtestValues, kPerfTestName);

}  // namespace batushin_i_striped_matrix_multiplication
