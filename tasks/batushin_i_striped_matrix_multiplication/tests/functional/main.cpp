#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "batushin_i_striped_matrix_multiplication/common/include/common.hpp"
#include "batushin_i_striped_matrix_multiplication/mpi/include/ops_mpi.hpp"
#include "batushin_i_striped_matrix_multiplication/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace batushin_i_striped_matrix_multiplication {

class BatushinIStripedMatrixMultiplicationFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<1>(params);
    expected_result_ = std::get<2>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_result_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_;
};

namespace {

InType CreateMatrices(size_t rows_a, size_t columns_a, const std::vector<double> &matrix_a, size_t rows_b,
                      size_t columns_b, const std::vector<double> &matrix_b) {
  return std::make_tuple(rows_a, columns_a, matrix_a, rows_b, columns_b, matrix_b);
}

TEST_P(BatushinIStripedMatrixMultiplicationFuncTests, MatrixMultiplication) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 15> kTestParam = {
    std::make_tuple("2x2_matrices", CreateMatrices(2, 2, {1.0, 2.0, 3.0, 4.0}, 2, 2, {5.0, 6.0, 7.0, 8.0}),
                    std::vector<double>({19.0, 22.0, 43.0, 50.0})),

    std::make_tuple("3x3_matrices",
                    CreateMatrices(3, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, 3, 3,
                                   {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0}),
                    std::vector<double>({30.0, 24.0, 18.0, 84.0, 69.0, 54.0, 138.0, 114.0, 90.0})),

    std::make_tuple("2x3_on_3x2",
                    CreateMatrices(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 3, 2, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0}),
                    std::vector<double>({58.0, 64.0, 139.0, 154.0})),

    std::make_tuple("row_vector_on_column_vector",
                    CreateMatrices(1, 4, {1.0, 2.0, 3.0, 4.0}, 4, 1, {5.0, 6.0, 7.0, 8.0}),
                    std::vector<double>({70.0})),

    std::make_tuple("identity_matrix",
                    CreateMatrices(3, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, 3, 3,
                                   {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}),
                    std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0})),

    std::make_tuple("zero_matrix", CreateMatrices(2, 2, {0.0, 0.0, 0.0, 0.0}, 2, 2, {1.0, 2.0, 3.0, 4.0}),
                    std::vector<double>({0.0, 0.0, 0.0, 0.0})),

    std::make_tuple("matrix_on_scalar",
                    CreateMatrices(3, 3, {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0}, 3, 3,
                                   {5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0}),
                    std::vector<double>({5.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 15.0})),

    std::make_tuple("float_numbers", CreateMatrices(2, 2, {1.5, 2.5, 3.5, 4.5}, 2, 2, {0.5, 1.5, 2.5, 3.5}),
                    std::vector<double>({7.0, 11.0, 13.0, 21.0})),

    std::make_tuple(
        "4x4_matrices",
        CreateMatrices(4, 4, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0}, 4,
                       4, {16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0}),
        std::vector<double>({80.0, 70.0, 60.0, 50.0, 240.0, 214.0, 188.0, 162.0, 400.0, 358.0, 316.0, 274.0, 560.0,
                             502.0, 444.0, 386.0})),

    std::make_tuple("1x1_matrices", CreateMatrices(1, 1, {7.5}, 1, 1, {3.2}), std::vector<double>({24.0})),

    std::make_tuple("5x3_on_3x2", CreateMatrices(5, 3, std::vector<double>(15, 1.0), 3, 2, std::vector<double>(6, 2.0)),
                    std::vector<double>(10, 6.0)),

    std::make_tuple("with_negatives", CreateMatrices(2, 2, {1.0, -2.0, -3.0, 4.0}, 2, 2, {-5.0, 6.0, 7.0, -8.0}),
                    std::vector<double>({-19.0, 22.0, 43.0, -50.0})),

    std::make_tuple("asymmetric_matrices",
                    CreateMatrices(3, 4, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, 4, 2,
                                   {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}),
                    std::vector<double>({50.0, 60.0, 114.0, 140.0, 178.0, 220.0})),

    std::make_tuple("large_numbers",
                    CreateMatrices(2, 2, {1000.0, 2000.0, 3000.0, 4000.0}, 2, 2, {0.001, 0.002, 0.003, 0.004}),
                    std::vector<double>({7.0, 10.0, 15.0, 22.0})),

    std::make_tuple("matrix_transpose",
                    CreateMatrices(3, 2, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 2, 3, {1.0, 3.0, 5.0, 2.0, 4.0, 6.0}),
                    std::vector<double>({5.0, 11.0, 17.0, 11.0, 25.0, 39.0, 17.0, 39.0, 61.0}))};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<BatushinIStripedMatrixMultiplicationMPI, InType>(
                                               kTestParam, PPC_SETTINGS_batushin_i_striped_matrix_multiplication),
                                           ppc::util::AddFuncTask<BatushinIStripedMatrixMultiplicationSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_batushin_i_striped_matrix_multiplication));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    BatushinIStripedMatrixMultiplicationFuncTests::PrintFuncTestName<BatushinIStripedMatrixMultiplicationFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationTests, BatushinIStripedMatrixMultiplicationFuncTests, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace batushin_i_striped_matrix_multiplication
