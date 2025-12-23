#include "batushin_i_striped_matrix_multiplication/seq/include/ops_seq.hpp"

#include <cstddef>
#include <tuple>
#include <vector>

#include "batushin_i_striped_matrix_multiplication/common/include/common.hpp"

namespace batushin_i_striped_matrix_multiplication {

BatushinIStripedMatrixMultiplicationSEQ::BatushinIStripedMatrixMultiplicationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BatushinIStripedMatrixMultiplicationSEQ::ValidationImpl() {
  const auto &input = GetInput();

  const size_t rows_a = std::get<0>(input);
  const size_t columns_a = std::get<1>(input);
  const auto &matrix_a = std::get<2>(input);

  const size_t rows_b = std::get<3>(input);
  const size_t columns_b = std::get<4>(input);
  const auto &matrix_b = std::get<5>(input);

  if (rows_a == 0 || columns_a == 0 || rows_b == 0 || columns_b == 0) {
    return false;
  }

  if (columns_a != rows_b) {
    return false;
  }

  if (matrix_a.size() != rows_a * columns_a) {
    return false;
  }

  if (matrix_b.size() != rows_b * columns_b) {
    return false;
  }

  return GetOutput().empty();
}

bool BatushinIStripedMatrixMultiplicationSEQ::PreProcessingImpl() {
  const auto &input = GetInput();

  const size_t rows_a = std::get<0>(input);
  const size_t columns_a = std::get<1>(input);
  const auto &matrix_a = std::get<2>(input);

  const size_t rows_b = std::get<3>(input);
  const size_t columns_b = std::get<4>(input);
  const auto &matrix_b = std::get<5>(input);

  return (columns_a == rows_b) && (matrix_a.size() == rows_a * columns_a) && (matrix_b.size() == rows_b * columns_b);
}

bool BatushinIStripedMatrixMultiplicationSEQ::RunImpl() {
  const auto &input = GetInput();

  const size_t rows_a = std::get<0>(input);
  const size_t columns_a = std::get<1>(input);
  const auto &matrix_a = std::get<2>(input);

  const size_t columns_b = std::get<4>(input);
  const auto &matrix_b = std::get<5>(input);

  auto &result = GetOutput();
  result.resize(rows_a * columns_b, 0.0);

  for (size_t i = 0; i < rows_a; i++) {
    for (size_t j = 0; j < columns_b; j++) {
      double sum = 0.0;
      for (size_t k = 0; k < columns_a; k++) {
        sum += matrix_a[(i * columns_a) + k] * matrix_b[(k * columns_b) + j];
      }
      result[(i * columns_b) + j] = sum;
    }
  }

  return true;
}

bool BatushinIStripedMatrixMultiplicationSEQ::PostProcessingImpl() {
  const auto &output = GetOutput();

  return !output.empty();
}

}  // namespace batushin_i_striped_matrix_multiplication
