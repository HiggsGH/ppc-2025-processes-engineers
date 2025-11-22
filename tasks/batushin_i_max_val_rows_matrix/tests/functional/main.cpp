#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "batushin_i_max_val_rows_matrix/common/include/common.hpp"
#include "batushin_i_max_val_rows_matrix/mpi/include/ops_mpi.hpp"
#include "batushin_i_max_val_rows_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace batushin_i_max_val_rows_matrix {

class BatushinIMaxValRowsMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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

InType CreateMatrix(size_t rows, size_t columns, const std::vector<double> &matrix) {
  return std::make_tuple(rows, columns, matrix);
}

TEST_P(BatushinIMaxValRowsMatrixFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 15> kTestParam = {
    std::make_tuple("3x3", CreateMatrix(3, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}),
                    std::vector<double>({3.0, 6.0, 9.0})),
    std::make_tuple("single_row", CreateMatrix(1, 4, {1.5, 2.7, 3.1, 2.9}), std::vector<double>({3.1})),
    std::make_tuple("with_negatives", CreateMatrix(3, 3, {-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0}),
                    std::vector<double>({-1.0, -4.0, -7.0})),
    std::make_tuple("same_values", CreateMatrix(3, 3, {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0}),
                    std::vector<double>({3.0, 3.0, 3.0})),
    std::make_tuple("large_matrix", CreateMatrix(100, 100, std::vector<double>(10000, 1.0)),
                    std::vector<double>(100, 1.0)),
    std::make_tuple("single_element", CreateMatrix(1, 1, {3.0}), std::vector<double>({3.0})),
    std::make_tuple("two_rows_three_columns", CreateMatrix(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
                    std::vector<double>({3.0, 6.0})),
    std::make_tuple(
        "three_proc_five_rows",
        CreateMatrix(5, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0}),
        std::vector<double>({3.0, 6.0, 9.0, 12.0, 15.0})),
    std::make_tuple("four_proc_seven_rows",
                    CreateMatrix(7, 2, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0}),
                    std::vector<double>({2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0})),
    std::make_tuple("two_by_one_matrix", CreateMatrix(2, 1, {5.0, 3.0}), std::vector<double>({5.0, 3.0})),
    std::make_tuple("one_row_many_columns", CreateMatrix(1, 10, {1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 5.0, 0.0}),
                    std::vector<double>({9})),
    std::make_tuple("many_rows_one_column", CreateMatrix(10, 1, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}),
                    std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0})),
    std::make_tuple("max_last", CreateMatrix(3, 3, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0}),
                    std::vector<double>({1.0, 1.0, 9.0})),
    std::make_tuple("max_is_first_elem", CreateMatrix(3, 3, {9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}),
                    std::vector<double>({9.0, 1.0, 1.0})),
    std::make_tuple("max_in_middle", CreateMatrix(3, 3, {1.0, 9.0, 1.0, 1.0, 9.0, 1.0, 1.0, 9.0, 1.0}),
                    std::vector<double>({9.0, 9.0, 9.0})),
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<BatushinIMaxValRowsMatrixMPI, InType>(
                                               kTestParam, PPC_SETTINGS_batushin_i_max_val_rows_matrix),
                                           ppc::util::AddFuncTask<BatushinIMaxValRowsMatrixSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_batushin_i_max_val_rows_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BatushinIMaxValRowsMatrixFuncTests::PrintFuncTestName<BatushinIMaxValRowsMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, BatushinIMaxValRowsMatrixFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace batushin_i_max_val_rows_matrix
