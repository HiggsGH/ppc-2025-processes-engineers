#include "batushin_i_striped_matrix_multiplication/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "batushin_i_striped_matrix_multiplication/common/include/common.hpp"

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

bool BatushinIStripedMatrixMultiplicationMPI::PreProcessingImpl() {
  return ValidationImpl();
}

namespace {

enum class MPITag { kMatrixB = 101 };

std::vector<int> ComputeBlockSizes(int total, int num_procs) {
  std::vector<int> sizes(num_procs, 0);
  const int block_size = total / num_procs;
  const int extra = total % num_procs;
  for (int i = 0; i < num_procs; ++i) {
    sizes[i] = block_size + (i < extra ? 1 : 0);
  }
  return sizes;
}

std::vector<int> ComputeBlockOffsets(const std::vector<int> &sizes) {
  std::vector<int> offsets(sizes.size(), 0);
  for (size_t idx = 1; idx < sizes.size(); ++idx) {
    offsets[idx] = offsets[idx - 1] + sizes[idx - 1];
  }
  return offsets;
}

void PerformLocalMultiplication(const std::vector<double> &local_a, const std::vector<double> &full_b,
                                std::vector<double> &local_c, int my_rows, int m, int p) {
  for (int i = 0; i < my_rows; ++i) {
    for (int j = 0; j < p; ++j) {
      double sum = 0.0;
      for (int k = 0; k < m; ++k) {
        sum += local_a[(i * m) + k] * full_b[(k * p) + j];
      }
      local_c[(i * p) + j] = sum;
    }
  }
}

}  // namespace

bool BatushinIStripedMatrixMultiplicationMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  const size_t rows_a = std::get<0>(input);
  const size_t cols_a = std::get<1>(input);
  const auto &matrix_a = std::get<2>(input);
  const size_t cols_b = std::get<4>(input);
  const auto &matrix_b = std::get<5>(input);

  int n = static_cast<int>(rows_a);
  int m = static_cast<int>(cols_a);
  int p = static_cast<int>(cols_b);

  std::vector<double> full_b;
  if (rank == 0) {
    full_b = matrix_b;
  } else {
    full_b.resize(static_cast<size_t>(m) * static_cast<size_t>(p));
    MPI_Recv(full_b.data(), static_cast<int>(full_b.size()), MPI_DOUBLE, 0, static_cast<int>(MPITag::kMatrixB),
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (rank == 0) {
    for (int dest = 1; dest < size; ++dest) {
      MPI_Send(matrix_b.data(), static_cast<int>(matrix_b.size()), MPI_DOUBLE, dest, static_cast<int>(MPITag::kMatrixB),
               MPI_COMM_WORLD);
    }
  }

  auto row_counts = ComputeBlockSizes(n, size);
  auto row_displs = ComputeBlockOffsets(row_counts);
  int my_rows = row_counts[rank];

  std::vector<double> local_a;
  if (my_rows > 0) {
    local_a.resize(static_cast<size_t>(my_rows) * static_cast<size_t>(m));
  }

  std::vector<int> sendcounts_a(size);
  std::vector<int> displs_a(size);
  for (int i = 0; i < size; ++i) {
    sendcounts_a[i] = row_counts[i] * m;
    displs_a[i] = row_displs[i] * m;
  }

  if (my_rows > 0) {
    MPI_Scatterv(matrix_a.data(), sendcounts_a.data(), displs_a.data(), MPI_DOUBLE, local_a.data(), my_rows * m,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(matrix_a.data(), sendcounts_a.data(), displs_a.data(), MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
  }

  std::vector<double> local_c;
  if (my_rows > 0) {
    local_c.resize(static_cast<size_t>(my_rows) * static_cast<size_t>(p), 0.0);
    PerformLocalMultiplication(local_a, full_b, local_c, my_rows, m, p);
  }

  std::vector<int> recvcounts(size);
  std::vector<int> recvdispls(size);
  for (int i = 0; i < size; ++i) {
    recvcounts[i] = row_counts[i] * p;
    recvdispls[i] = row_displs[i] * p;
  }

  std::vector<double> result;
  if (rank == 0) {
    result.resize(static_cast<size_t>(n) * static_cast<size_t>(p));
  }

  if (my_rows > 0) {
    MPI_Gatherv(local_c.data(), my_rows * p, MPI_DOUBLE, result.data(), recvcounts.data(), recvdispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Gatherv(nullptr, 0, MPI_DOUBLE, result.data(), recvcounts.data(), recvdispls.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
  }

  int total_size = n * p;
  if (rank != 0) {
    result.resize(static_cast<size_t>(total_size));
  }
  MPI_Bcast(result.data(), total_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = std::move(result);
  return true;
}

bool BatushinIStripedMatrixMultiplicationMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace batushin_i_striped_matrix_multiplication
