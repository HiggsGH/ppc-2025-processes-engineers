#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "batushin_i_quick_sort_with_simple_merge/common/include/common.hpp"
#include "batushin_i_quick_sort_with_simple_merge/mpi/include/ops_mpi.hpp"
#include "batushin_i_quick_sort_with_simple_merge/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace batushin_i_quick_sort_with_simple_merge {

class BatushinIQuickSortWithSimpleMergeFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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
    return expected_result_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_;
};

namespace {

InType CreateInput(std::vector<int> data) {
  return data;
}

std::vector<int> SortedCopy(const std::vector<int> &v) {
  auto res = v;
  std::ranges::sort(res, std::less<>());
  return res;
}

TEST_P(BatushinIQuickSortWithSimpleMergeFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 15> kTestParam = {
    std::make_tuple("empty", CreateInput({}), std::vector<int>({})),
    std::make_tuple("single", CreateInput({42}), std::vector<int>({42})),
    std::make_tuple("two_desc", CreateInput({5, 3}), std::vector<int>({3, 5})),
    std::make_tuple("reverse_5", CreateInput({5, 4, 3, 2, 1}), SortedCopy({5, 4, 3, 2, 1})),
    std::make_tuple("random_7", CreateInput({3, 1, 4, 1, 5, 9, 2}), SortedCopy({3, 1, 4, 1, 5, 9, 2})),
    std::make_tuple("all_same", CreateInput({7, 7, 7, 7}), std::vector<int>({7, 7, 7, 7})),
    std::make_tuple("with_negatives", CreateInput({-1, -3, 2, 0, -5}), SortedCopy({-1, -3, 2, 0, -5})),
    std::make_tuple("large_asc", CreateInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
                    std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})),
    std::make_tuple("large_desc", CreateInput({10, 9, 8, 7, 6, 5, 4, 3, 2, 1}),
                    SortedCopy({10, 9, 8, 7, 6, 5, 4, 3, 2, 1})),
    std::make_tuple("duplicates_mixed", CreateInput({2, 1, 2, 3, 1, 3}), SortedCopy({2, 1, 2, 3, 1, 3})),
    std::make_tuple("zigzag_6", CreateInput({1, 5, 2, 4, 3, 6}), SortedCopy({1, 5, 2, 4, 3, 6})),
    std::make_tuple("large_sorted_20",
                    CreateInput({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}),
                    std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19})),
    std::make_tuple("large_reverse_17", CreateInput({16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}),
                    std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})),
    std::make_tuple("edge_case_16", CreateInput({15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}),
                    std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})),
    std::make_tuple("already_sorted", CreateInput({-5, -2, 0, 3, 7}), std::vector<int>({-5, -2, 0, 3, 7}))};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<BatushinIQuickSortWithSimpleMergeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_batushin_i_quick_sort_with_simple_merge),
                                           ppc::util::AddFuncTask<BatushinIQuickSortWithSimpleMergeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_batushin_i_quick_sort_with_simple_merge));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    BatushinIQuickSortWithSimpleMergeFuncTests::PrintFuncTestName<BatushinIQuickSortWithSimpleMergeFuncTests>;

INSTANTIATE_TEST_SUITE_P(QuickSortWithSimpleMergeTests, BatushinIQuickSortWithSimpleMergeFuncTests, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace batushin_i_quick_sort_with_simple_merge
