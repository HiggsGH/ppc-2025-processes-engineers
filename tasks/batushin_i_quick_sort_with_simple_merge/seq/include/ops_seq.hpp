#pragma once

#include "batushin_i_quick_sort_with_simple_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace batushin_i_quick_sort_with_simple_merge {

class BatushinIQuickSortWithSimpleMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BatushinIQuickSortWithSimpleMergeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace batushin_i_quick_sort_with_simple_merge
