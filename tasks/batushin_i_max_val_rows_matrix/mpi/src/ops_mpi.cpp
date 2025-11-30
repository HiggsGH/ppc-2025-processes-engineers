#include "batushin_i_max_val_rows_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "batushin_i_max_val_rows_matrix/common/include/common.hpp"

namespace batushin_i_max_val_rows_matrix {

BatushinIMaxValRowsMatrixMPI::BatushinIMaxValRowsMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool BatushinIMaxValRowsMatrixMPI::ValidationImpl() {
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

bool BatushinIMaxValRowsMatrixMPI::PreProcessingImpl() {
  const auto &input = GetInput();
  const size_t rows = std::get<0>(input);
  const size_t columns = std::get<1>(input);
  const auto &matrix = std::get<2>(input);

  return (rows > 0) && (columns > 0) && (matrix.size() == rows * columns);
}

namespace {

std::pair<size_t, size_t> GetRowRange(int rank, int proc, size_t rows) {
  size_t base_rows = rows / proc;
  size_t extra_rows = rows % proc;

  size_t start_row = (rank * base_rows) + std::min<size_t>(rank, extra_rows);
  size_t end_row = start_row + base_rows + (std::cmp_less(rank, extra_rows) ? 1 : 0);

  return {start_row, end_row};
}

std::vector<double> CalcLocalMax(size_t start_row, size_t end_row, size_t columns, const std::vector<double> &matrix) {
  std::vector<double> loc_max;

  for (size_t i = start_row; i < end_row; i++) {
    double max_val = matrix[i * columns];
    for (size_t j = 1; j < columns; j++) {
      double val = matrix[(i * columns) + j];
      max_val = std::max(val, max_val);
    }
    loc_max.push_back(max_val);
  }

  return loc_max;
}

void CollectResults(int rank, int proc, size_t rows, const std::vector<double> &loc_max, size_t start_row,
                    std::vector<double> &res) {
  if (rank == 0) {
    for (size_t i = 0; i < loc_max.size(); i++) {
      res[start_row + i] = loc_max[i];
    }

    for (int src = 1; src < proc; src++) {
      auto [src_start, src_end] = GetRowRange(src, proc, rows);
      size_t src_size = src_end - src_start;

      std::vector<double> recv_buf(src_size);
      MPI_Recv(recv_buf.data(), static_cast<int>(src_size), MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (size_t i = 0; i < src_size; i++) {
        res[src_start + i] = recv_buf[i];
      }
    }
  } else {
    MPI_Send(loc_max.data(), static_cast<int>(loc_max.size()), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
}

void SynchronizationResult(int rank, std::vector<double> &res) {
  if (rank == 0) {
    int res_size = static_cast<int>(res.size());
    MPI_Bcast(&res_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(res.data(), res_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    int res_size = 0;
    MPI_Bcast(&res_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    res.resize(res_size);
    MPI_Bcast(res.data(), res_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

}  // namespace

bool BatushinIMaxValRowsMatrixMPI::RunImpl() {
  int rank = 0;
  int proc = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc);

  size_t rows = std::get<0>(GetInput());
  size_t columns = std::get<1>(GetInput());
  const auto &matrix = std::get<2>(GetInput());

  auto [start_row, end_row] = GetRowRange(rank, proc, rows);
  auto loc_max = CalcLocalMax(start_row, end_row, columns, matrix);

  std::vector<double> res;
  if (rank == 0) {
    res.resize(rows);
  }

  CollectResults(rank, proc, rows, loc_max, start_row, res);
  SynchronizationResult(rank, res);

  GetOutput() = res;
  return true;
}

bool BatushinIMaxValRowsMatrixMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace batushin_i_max_val_rows_matrix
