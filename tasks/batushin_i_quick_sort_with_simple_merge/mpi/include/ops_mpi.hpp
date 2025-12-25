#pragma once

#include "batushin_i_quick_sort_with_simple_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace batushin_i_quick_sort_with_simple_merge {

class BatushinIQuickSortWithSimpleMergeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BatushinIQuickSortWithSimpleMergeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace batushin_i_quick_sort_with_simple_merge
