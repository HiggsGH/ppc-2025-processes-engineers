#pragma once

#include "batushin_i_max_val_rows_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace batushin_i_max_val_rows_matrix {

class BatushinIMaxValRowsMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BatushinIMaxValRowsMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace batushin_i_max_val_rows_matrix
