#include "batushin_i_max_val_rows_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "batushin_i_max_val_rows_matrix/common/include/common.hpp"

namespace batushin_i_max_val_rows_matrix {

BatushinIMaxValRowsMatrixSEQ::BatushinIMaxValRowsMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BatushinIMaxValRowsMatrixSEQ::ValidationImpl() {
  const auto &input = GetInput();
  const size_t rows = std::get<0>(input);
  const size_t columns = std::get<1>(input);
  const auto &matrix = std::get<2>(input);

  if (rows == 0 || columns == 0) {
    return false;
  }

  if (matrix.size() != rows * columns) {
    return false;
  }

  return GetOutput().empty();
}

bool BatushinIMaxValRowsMatrixSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  const size_t rows = std::get<0>(input);
  const size_t columns = std::get<1>(input);
  const auto &matrix = std::get<2>(input);

  return (rows > 0) && (columns > 0) && (matrix.size() == rows * columns);
}

bool BatushinIMaxValRowsMatrixSEQ::RunImpl() {
  const auto &input = GetInput();
  const size_t rows = std::get<0>(input);
  const size_t columns = std::get<1>(input);
  const auto &matrix = std::get<2>(input);
  auto &res = GetOutput();

  res.resize(rows);

  for (size_t i = 0; i < rows; i++) {
    double max_val = matrix[i * columns];

    for (size_t j = 1; j < columns; j++) {
      const double curr_val = matrix[(i * columns) + j];
      max_val = std::max(curr_val, max_val);
    }

    res[i] = max_val;
  }

  return true;
}

bool BatushinIMaxValRowsMatrixSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace batushin_i_max_val_rows_matrix
