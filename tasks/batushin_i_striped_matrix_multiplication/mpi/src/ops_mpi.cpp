#include "batushin_i_striped_matrix_multiplication/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
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

namespace {

std::pair<size_t, size_t> CalculateRowDistribution(int rank, int size, size_t n, size_t &rows_per_proc,
                                                   size_t &extra_rows) {
  rows_per_proc = n / size;
  extra_rows = n % size;

  size_t my_rows = rows_per_proc + (std::cmp_less(rank, extra_rows) ? 1 : 0);
  size_t my_start = (rank * rows_per_proc) + std::min<size_t>(rank, extra_rows);

  return {my_rows, my_start};
}

void FillLocalPart(size_t my_rows, size_t m, size_t my_start, const std::vector<double> &matrix_a,
                   std::vector<double> &local_a) {
  for (size_t i = 0; i < my_rows; ++i) {
    for (size_t j = 0; j < m; ++j) {
      local_a[(i * m) + j] = matrix_a[((my_start + i) * m) + j];
    }
  }
}

void FillBufferForProcess(int dest, size_t m, size_t rows_per_proc, size_t extra_rows,
                          const std::vector<double> &matrix_a, std::vector<double> &buffer) {
  size_t dest_rows = rows_per_proc + (std::cmp_less(dest, extra_rows) ? 1 : 0);
  if (dest_rows == 0) {
    return;
  }

  size_t dest_start = (dest * rows_per_proc) + std::min<size_t>(dest, extra_rows);

  buffer.resize(dest_rows * m);
  for (size_t i = 0; i < dest_rows; ++i) {
    for (size_t j = 0; j < m; ++j) {
      buffer[(i * m) + j] = matrix_a[((dest_start + i) * m) + j];
    }
  }
}

void SendToProcesses(int size, size_t m, size_t rows_per_proc, size_t extra_rows, const std::vector<double> &matrix_a) {
  for (int dest = 1; dest < size; ++dest) {
    std::vector<double> buffer;
    FillBufferForProcess(dest, m, rows_per_proc, extra_rows, matrix_a, buffer);
    if (!buffer.empty()) {
      MPI_Send(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
    }
  }
}

std::vector<double> DistributeMatrixA(int rank, int size, size_t my_rows, size_t my_start, size_t m,
                                      size_t rows_per_proc, size_t extra_rows, const std::vector<double> &matrix_a) {
  std::vector<double> local_a(my_rows * m);

  if (rank == 0) {
    FillLocalPart(my_rows, m, my_start, matrix_a, local_a);
    SendToProcesses(size, m, rows_per_proc, extra_rows, matrix_a);
  } else if (my_rows > 0) {
    MPI_Recv(local_a.data(), static_cast<int>(local_a.size()), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  return local_a;
}

std::vector<double> BroadcastMatrixB(int rank, size_t m, size_t p, const std::vector<double> &matrix_b) {
  std::vector<double> local_b(m * p);

  if (rank == 0) {
    local_b = matrix_b;
  }

  MPI_Bcast(local_b.data(), static_cast<int>(m * p), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return local_b;
}

std::vector<double> LocalMatrixMultiplication(const std::vector<double> &local_a, const std::vector<double> &local_b,
                                              size_t my_rows, size_t m, size_t p) {
  std::vector<double> local_c(my_rows * p, 0.0);

  for (size_t i = 0; i < my_rows; ++i) {
    for (size_t j = 0; j < p; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < m; ++k) {
        sum += local_a[(i * m) + k] * local_b[(k * p) + j];
      }
      local_c[(i * p) + j] = sum;
    }
  }

  return local_c;
}

void FillLocalResultPart(size_t my_rows, size_t my_start, size_t p, const std::vector<double> &local_c,
                         std::vector<double> &result) {
  for (size_t i = 0; i < my_rows; ++i) {
    for (size_t j = 0; j < p; ++j) {
      result[((my_start + i) * p) + j] = local_c[(i * p) + j];
    }
  }
}

void ReceiveFromProcess(int src, size_t p, size_t rows_per_proc, size_t extra_rows, std::vector<double> &result) {
  size_t src_rows = rows_per_proc + (std::cmp_less(src, extra_rows) ? 1 : 0);
  if (src_rows == 0) {
    return;
  }

  size_t src_start = (src * rows_per_proc) + std::min<size_t>(src, extra_rows);

  std::vector<double> buffer(src_rows * p);
  MPI_Recv(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for (size_t i = 0; i < src_rows; ++i) {
    for (size_t j = 0; j < p; ++j) {
      result[((src_start + i) * p) + j] = buffer[(i * p) + j];
    }
  }
}

void ReceiveFromAllProcesses(int size, size_t p, size_t rows_per_proc, size_t extra_rows, std::vector<double> &result) {
  for (int src = 1; src < size; ++src) {
    ReceiveFromProcess(src, p, rows_per_proc, extra_rows, result);
  }
}

void GatherResults(int rank, int size, const std::vector<double> &local_c, size_t my_rows, size_t my_start, size_t n,
                   size_t p, size_t rows_per_proc, size_t extra_rows, std::vector<double> &result) {
  if (rank == 0) {
    result.resize(n * p);
    FillLocalResultPart(my_rows, my_start, p, local_c, result);
    ReceiveFromAllProcesses(size, p, rows_per_proc, extra_rows, result);
  } else if (my_rows > 0) {
    MPI_Send(local_c.data(), static_cast<int>(local_c.size()), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
}

void SynchronizeResult(int rank, std::vector<double> &result) {
  if (rank == 0) {
    int result_size = static_cast<int>(result.size());

    MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (result_size > 0) {
      MPI_Bcast(result.data(), result_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  } else {
    int result_size = 0;
    MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (result_size > 0) {
      result.resize(result_size);
      MPI_Bcast(result.data(), result_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
      result.clear();
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
  const size_t columns_a = std::get<1>(input);
  const auto &matrix_a = std::get<2>(input);
  const size_t rows_b = std::get<3>(input);
  const size_t columns_b = std::get<4>(input);
  const auto &matrix_b = std::get<5>(input);

  std::array<unsigned long long, 4> dims{};
  if (rank == 0) {
    dims[0] = rows_a;
    dims[1] = columns_a;
    dims[2] = rows_b;
    dims[3] = columns_b;
  }

  MPI_Bcast(dims.data(), 4, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  const size_t n = static_cast<size_t>(dims[0]);
  const size_t m = static_cast<size_t>(dims[1]);
  const size_t p = static_cast<size_t>(dims[3]);

  if (n == 0 || m == 0 || p == 0) {
    if (rank == 0) {
      GetOutput() = std::vector<double>();
    }
    return true;
  }

  size_t rows_per_proc = 0;
  size_t extra_rows = 0;
  auto [my_rows, my_start] = CalculateRowDistribution(rank, size, n, rows_per_proc, extra_rows);

  auto local_a = DistributeMatrixA(rank, size, my_rows, my_start, m, rows_per_proc, extra_rows, matrix_a);

  auto local_b = BroadcastMatrixB(rank, m, p, matrix_b);

  auto local_c = LocalMatrixMultiplication(local_a, local_b, my_rows, m, p);

  std::vector<double> result;
  GatherResults(rank, size, local_c, my_rows, my_start, n, p, rows_per_proc, extra_rows, result);

  SynchronizeResult(rank, result);

  GetOutput() = result;

  return true;
}

bool BatushinIStripedMatrixMultiplicationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace batushin_i_striped_matrix_multiplication
