#include "batushin_i_striped_matrix_multiplication/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
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

enum class MPITag : std::uint8_t { kMatrixB = 101 };

std::vector<int> ComputeBlockSizes(int total, int num_procs) {
  std::vector<int> sizes(num_procs, 0);
  int base = total / num_procs;
  int extra = total % num_procs;
  for (int i = 0; i < num_procs; ++i) {
    sizes[i] = base + (i < extra ? 1 : 0);
  }
  return sizes;
}

std::vector<int> ComputeBlockOffsets(const std::vector<int> &sizes) {
  std::vector<int> offsets(sizes.size(), 0);
  for (size_t i = 1; i < sizes.size(); ++i) {
    offsets[i] = offsets[i - 1] + sizes[i - 1];
  }
  return offsets;
}

bool RunSequentialFallback(int rank, size_t rows_a, size_t cols_a, size_t cols_b, const std::vector<double> &matrix_a,
                           const std::vector<double> &matrix_b, std::vector<double> &output) {
  if (rank == 0) {
    output.resize(rows_a * cols_b, 0.0);
    for (size_t i = 0; i < rows_a; ++i) {
      for (size_t j = 0; j < cols_b; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < cols_a; ++k) {
          sum += matrix_a[(i * cols_a) + k] * matrix_b[(k * cols_b) + j];
        }
        output[(i * cols_b) + j] = sum;
      }
    }
  }

  int total_size = (rank == 0) ? static_cast<int>(output.size()) : 0;
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    output.resize(total_size);
  }
  if (total_size > 0) {
    MPI_Bcast(output.data(), total_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  return true;
}

void DistributeMatrixAFromRoot(int size, int m, const std::vector<double> &matrix_a, const std::vector<int> &row_counts,
                               const std::vector<int> &row_displs, std::vector<double> &local_a) {
  std::vector<int> sendcounts_a(size);
  std::vector<int> displs_a(size);
  for (int i = 0; i < size; ++i) {
    sendcounts_a[i] = row_counts[i] * m;
    displs_a[i] = row_displs[i] * m;
  }

  int my_rows = (local_a.empty()) ? 0 : static_cast<int>(local_a.size()) / m;
  if (my_rows > 0) {
    MPI_Scatterv(matrix_a.data(), sendcounts_a.data(), displs_a.data(), MPI_DOUBLE, local_a.data(), my_rows * m,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(matrix_a.data(), sendcounts_a.data(), displs_a.data(), MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
  }
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<double>> DistributeMatrixA(
    int rank, int size, int n, int m, const std::vector<double> &matrix_a) {
  auto row_counts = ComputeBlockSizes(n, size);
  auto row_displs = ComputeBlockOffsets(row_counts);
  int my_rows = row_counts[rank];

  std::vector<double> local_a;
  if (my_rows > 0) {
    local_a.resize(static_cast<size_t>(my_rows) * static_cast<size_t>(m));
  }

  DistributeMatrixAFromRoot(size, m, matrix_a, row_counts, row_displs, local_a);
  return {row_counts, row_displs, local_a};
}

std::vector<double> ExtractColumnBlock(int m, int p, int dest_col_start, int dest_col_count,
                                       const std::vector<double> &matrix_b) {
  std::vector<double> buf(static_cast<size_t>(m) * static_cast<size_t>(dest_col_count));
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < dest_col_count; ++col) {
      int global_col = dest_col_start + col;
      buf[(static_cast<size_t>(row) * static_cast<size_t>(dest_col_count)) + static_cast<size_t>(col)] =
          matrix_b[(static_cast<size_t>(row) * static_cast<size_t>(p)) + static_cast<size_t>(global_col)];
    }
  }
  return buf;
}

void DistributeMatrixBFromRoot(int size, int m, int p, const std::vector<double> &matrix_b,
                               const std::vector<int> &col_counts, const std::vector<int> &col_displs,
                               std::vector<double> &current_b, int &current_cols) {
  for (int dest = 0; dest < size; ++dest) {
    if (col_counts[dest] > 0) {
      std::vector<double> buf = ExtractColumnBlock(m, p, col_displs[dest], col_counts[dest], matrix_b);
      if (dest == 0) {
        current_b = std::move(buf);
        current_cols = col_counts[0];
      } else {
        MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, dest, 100, MPI_COMM_WORLD);
      }
    } else {
      if (dest == 0) {
        current_cols = 0;
      } else {
        MPI_Send(nullptr, 0, MPI_DOUBLE, dest, 100, MPI_COMM_WORLD);
      }
    }
  }
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<double>, int> DistributeMatrixB(
    int rank, int size, int m, int p, const std::vector<double> &matrix_b) {
  auto col_counts = ComputeBlockSizes(p, size);
  auto col_displs = ComputeBlockOffsets(col_counts);

  std::vector<double> current_b;
  int current_cols = 0;

  if (rank == 0) {
    DistributeMatrixBFromRoot(size, m, p, matrix_b, col_counts, col_displs, current_b, current_cols);
  } else {
    if (col_counts[rank] > 0) {
      current_b.resize(static_cast<size_t>(m) * static_cast<size_t>(col_counts[rank]));
      MPI_Recv(current_b.data(), static_cast<int>(current_b.size()), MPI_DOUBLE, 0, 100, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      current_cols = col_counts[rank];
    } else {
      MPI_Recv(nullptr, 0, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      current_cols = 0;
    }
  }

  return {col_counts, col_displs, current_b, current_cols};
}

void PerformLocalComputation(int my_rows, int current_cols, int m, int p, int stripe_offset,
                             const std::vector<double> &local_a, const std::vector<double> &current_b,
                             std::vector<double> &local_c) {
  for (int i = 0; i < my_rows; ++i) {
    for (int j = 0; j < current_cols; ++j) {
      double sum = 0.0;
      for (int k = 0; k < m; ++k) {
        sum += local_a[(static_cast<size_t>(i) * static_cast<size_t>(m)) + static_cast<size_t>(k)] *
               current_b[static_cast<size_t>(k) + (static_cast<size_t>(j) * static_cast<size_t>(m))];
      }
      local_c[(static_cast<size_t>(i) * static_cast<size_t>(p)) + static_cast<size_t>(stripe_offset + j)] = sum;
    }
  }
}

std::pair<std::vector<double>, int> ShiftMatrixB(int rank, int size, int m, std::vector<double> &current_b,
                                                 int current_cols) {
  int next = (rank + 1) % size;
  int prev = (rank - 1 + size) % size;

  int send_cols = current_cols;
  int recv_cols = 0;
  MPI_Sendrecv(&send_cols, 1, MPI_INT, next, 200, &recv_cols, 1, MPI_INT, prev, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<double> recv_buffer;
  if (recv_cols > 0) {
    recv_buffer.resize(static_cast<size_t>(m) * static_cast<size_t>(recv_cols));
  }

  int send_count = send_cols * m;
  int recv_count = recv_cols * m;
  const double *send_ptr = (send_count > 0 && !current_b.empty()) ? current_b.data() : nullptr;
  double *recv_ptr = (recv_count > 0) ? recv_buffer.data() : nullptr;

  MPI_Sendrecv(send_ptr, send_count, MPI_DOUBLE, next, 201, recv_ptr, recv_count, MPI_DOUBLE, prev, 201, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  if (recv_cols > 0) {
    return {std::move(recv_buffer), recv_cols};
  }
  return {{}, 0};
}

std::vector<double> ComputeWithCyclicShift(int rank, int size, int m, int p, const std::vector<double> &local_a,
                                           std::vector<double> current_b, int current_cols,
                                           const std::vector<int> &col_displs) {
  int my_rows = (local_a.empty()) ? 0 : static_cast<int>(local_a.size()) / m;
  std::vector<double> local_c;
  if (my_rows > 0) {
    local_c.resize(static_cast<size_t>(my_rows) * static_cast<size_t>(p), 0.0);
  }

  int stripe_owner = rank;
  for (int step = 0; step < size; ++step) {
    if (my_rows > 0 && current_cols > 0 && !current_b.empty()) {
      int stripe_offset = (static_cast<size_t>(stripe_owner) < col_displs.size()) ? col_displs[stripe_owner] : 0;
      PerformLocalComputation(my_rows, current_cols, m, p, stripe_offset, local_a, current_b, local_c);
    }

    if (step == size - 1) {
      break;
    }

    auto [new_b, new_cols] = ShiftMatrixB(rank, size, m, current_b, current_cols);
    current_b = std::move(new_b);
    current_cols = new_cols;
    stripe_owner = (stripe_owner - 1 + size) % size;
  }

  return local_c;
}

void BroadcastFinalResult(int rank, int size, int n, int p, const std::vector<int> &row_counts,
                          const std::vector<int> &row_displs, const std::vector<double> &local_c,
                          std::vector<double> &result) {
  std::vector<int> result_counts(size);
  std::vector<int> result_displs(size);
  for (int i = 0; i < size; ++i) {
    result_counts[i] = row_counts[i] * p;
    result_displs[i] = row_displs[i] * p;
  }

  if (rank == 0) {
    result.resize(static_cast<size_t>(n) * static_cast<size_t>(p));
  }

  int my_rows = (local_c.empty()) ? 0 : static_cast<int>(local_c.size()) / p;
  int local_result_elements = my_rows * p;
  const double *local_result_ptr = (my_rows > 0) ? local_c.data() : nullptr;

  MPI_Gatherv(local_result_ptr, local_result_elements, MPI_DOUBLE, result.data(), result_counts.data(),
              result_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int total_size = n * p;
  if (rank != 0) {
    result.resize(static_cast<size_t>(total_size));
  }
  MPI_Bcast(result.data(), total_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool RunStripedScheme(int rank, int size, size_t rows_a, size_t cols_a, size_t cols_b,
                      const std::vector<double> &matrix_a, const std::vector<double> &matrix_b,
                      std::vector<double> &output) {
  const int n = static_cast<int>(rows_a);
  const int m = static_cast<int>(cols_a);
  const int p = static_cast<int>(cols_b);

  auto [row_counts, row_displs, local_a] = DistributeMatrixA(rank, size, n, m, matrix_a);
  auto [col_counts, col_displs, current_b, current_cols] = DistributeMatrixB(rank, size, m, p, matrix_b);
  auto local_c = ComputeWithCyclicShift(rank, size, m, p, local_a, current_b, current_cols, col_displs);

  BroadcastFinalResult(rank, size, n, p, row_counts, row_displs, local_c, output);
  return true;
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

  std::vector<double> output;

  if (std::cmp_greater(size, rows_a) || std::cmp_greater(size, cols_b) || size <= 4) {
    RunSequentialFallback(rank, rows_a, cols_a, cols_b, matrix_a, matrix_b, output);
  } else {
    RunStripedScheme(rank, size, rows_a, cols_a, cols_b, matrix_a, matrix_b, output);
  }

  GetOutput() = std::move(output);
  return true;
}

bool BatushinIStripedMatrixMultiplicationMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace batushin_i_striped_matrix_multiplication
