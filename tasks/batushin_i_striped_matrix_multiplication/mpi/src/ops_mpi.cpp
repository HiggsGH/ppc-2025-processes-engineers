#include "batushin_i_striped_matrix_multiplication/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "batushin_i_striped_matrix_multiplication/common/include/common.hpp"
#include "util/include/util.hpp"

namespace batushin_i_striped_matrix_multiplication {

BatushinIStripedMatrixMultiplicationMPI::BatushinIStripedMatrixMultiplicationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BatushinIStripedMatrixMultiplicationMPI::ValidationImpl() {
  const auto &input = GetInput();

  const size_t rows_a = std::get<0>(input);
  const size_t columns_a = std::get<1>(input);
  const auto &matrix_a = std::get<2>(input);

  const size_t rows_b = std::get<3>(input);
  const size_t columns_b = std::get<4>(input);
  const auto &matrix_b = std::get<5>(input);

  if (columns_a != rows_b) {
    return false;
  }
  if (matrix_a.size() != rows_a * columns_a) {
    return false;
  }
  if (matrix_b.size() != rows_b * columns_b) {
    return false;
  }

  return true;
}

bool BatushinIStripedMatrixMultiplicationMPI::PreProcessingImpl() {
  return ValidationImpl();
}

bool BatushinIStripedMatrixMultiplicationMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();

  const size_t rows_a = std::get<0>(input);
  const size_t columns_a = std::get<1>(input);
  const auto &matrix_a = std::get<2>(input);
  const size_t rows_b = std::get<3>(input);
  const size_t columns_b = std::get<4>(input);
  const auto &matrix_b = std::get<5>(input);

  size_t dims[4] = {rows_a, columns_a, rows_b, columns_b};
  MPI_Bcast(dims, 4, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  const size_t n = dims[0];
  const size_t m = dims[1];
  const size_t p = dims[3];

  size_t rows_per_proc = n / size;
  size_t extra_rows = n % size;

  size_t my_rows = rows_per_proc + (rank < static_cast<int>(extra_rows) ? 1 : 0);
  size_t my_start = rank * rows_per_proc + std::min<size_t>(rank, extra_rows);

  std::vector<double> local_a(my_rows * m);
  std::vector<double> local_c(my_rows * p, 0.0);

  if (rank == 0) {
    for (size_t i = 0; i < my_rows; i++) {
      for (size_t j = 0; j < m; j++) {
        local_a[i * m + j] = matrix_a[(my_start + i) * m + j];
      }
    }

    for (int dest = 1; dest < size; dest++) {
      size_t dest_rows = rows_per_proc + (dest < static_cast<int>(extra_rows) ? 1 : 0);
      if (dest_rows > 0) {
        size_t dest_start = dest * rows_per_proc + std::min<size_t>(dest, extra_rows);

        std::vector<double> buffer(dest_rows * m);
        for (size_t i = 0; i < dest_rows; i++) {
          for (size_t j = 0; j < m; j++) {
            buffer[i * m + j] = matrix_a[(dest_start + i) * m + j];
          }
        }

        MPI_Send(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
      }
    }
  } else if (my_rows > 0) {
    MPI_Recv(local_a.data(), static_cast<int>(local_a.size()), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  std::vector<double> local_b(m * p);
  if (rank == 0) {
    local_b = matrix_b;
  }
  MPI_Bcast(local_b.data(), static_cast<int>(m * p), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (size_t i = 0; i < my_rows; i++) {
    for (size_t j = 0; j < p; j++) {
      double sum = 0.0;
      for (size_t k = 0; k < m; k++) {
        sum += local_a[i * m + k] * local_b[k * p + j];
      }
      local_c[i * p + j] = sum;
    }
  }

  if (rank == 0) {
    std::vector<double> result(n * p);

    for (size_t i = 0; i < my_rows; i++) {
      for (size_t j = 0; j < p; j++) {
        result[(my_start + i) * p + j] = local_c[i * p + j];
      }
    }

    for (int src = 1; src < size; src++) {
      size_t src_rows = rows_per_proc + (src < static_cast<int>(extra_rows) ? 1 : 0);
      if (src_rows > 0) {
        size_t src_start = src * rows_per_proc + std::min<size_t>(src, extra_rows);

        std::vector<double> buffer(src_rows * p);
        MPI_Recv(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (size_t i = 0; i < src_rows; i++) {
          for (size_t j = 0; j < p; j++) {
            result[(src_start + i) * p + j] = buffer[i * p + j];
          }
        }
      }
    }

    GetOutput() = result;
  } else if (my_rows > 0) {
    MPI_Send(local_c.data(), static_cast<int>(local_c.size()), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    int result_size = static_cast<int>(n * p);
    MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(GetOutput().data(), result_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    int result_size = 0;
    MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<double> result(result_size);
    if (result_size > 0) {
      MPI_Bcast(result.data(), result_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    GetOutput() = result;
  }

  return true;
}

bool BatushinIStripedMatrixMultiplicationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace batushin_i_striped_matrix_multiplication
