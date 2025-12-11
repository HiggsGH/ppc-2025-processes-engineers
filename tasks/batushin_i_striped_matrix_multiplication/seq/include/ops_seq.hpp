#pragma once

#include "batushin_i_striped_matrix_multiplication/common/include/common.hpp"
#include "task/include/task.hpp"

namespace batushin_i_striped_matrix_multiplication {

class BatushinIStripedMatrixMultiplicationSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BatushinIStripedMatrixMultiplicationSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace batushin_i_striped_matrix_multiplication
