#include "batushin_i_quick_sort_with_simple_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <iterator>
#include <stack>
#include <vector>

#include "batushin_i_quick_sort_with_simple_merge/common/include/common.hpp"

namespace batushin_i_quick_sort_with_simple_merge {

BatushinIQuickSortWithSimpleMergeMPI::BatushinIQuickSortWithSimpleMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool BatushinIQuickSortWithSimpleMergeMPI::ValidationImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  return initialized != 0;
}

bool BatushinIQuickSortWithSimpleMergeMPI::PreProcessingImpl() {
  return true;
}

namespace {

// Простая сортировка без сложной логики
void sortSmallArray(std::vector<int> &data, int begin, int end) {
  for (int i = begin + 1; i <= end; ++i) {
    int key = data[i];
    int j = i - 1;
    while (j >= begin && data[j] > key) {
      data[j + 1] = data[j];
      --j;
    }
    data[j + 1] = key;
  }
}

void iterativeQuickSort(std::vector<int> &data) {
  if (data.size() <= 1) {
    return;
  }

  const int threshold = 16;
  struct Segment {
    int begin, end;
  };
  std::stack<Segment> stk;
  stk.push({0, static_cast<int>(data.size() - 1)});

  while (!stk.empty()) {
    Segment seg = stk.top();
    stk.pop();
    if (seg.begin >= seg.end) {
      continue;
    }

    if (seg.end - seg.begin + 1 <= threshold) {
      sortSmallArray(data, seg.begin, seg.end);
      continue;
    }

    int mid = seg.begin + ((seg.end - seg.begin) / 2);
    std::swap(data[mid], data[seg.begin]);
    int pivot = data[seg.begin];

    int i = seg.begin;
    int j = seg.end;
    while (i <= j) {
      while (data[i] < pivot) {
        ++i;
      }
      while (data[j] > pivot) {
        --j;
      }
      if (i <= j) {
        std::swap(data[i], data[j]);
        ++i;
        --j;
      }
    }

    if (seg.begin < j) {
      stk.push({seg.begin, j});
    }
    if (i < seg.end) {
      stk.push({i, seg.end});
    }
  }
}

std::pair<int, int> computeLocalRange(int rank, int size, int total) {
  int base = total / size;
  int extra = total % size;
  int start = (rank * base) + std::min(rank, extra);
  int end = start + base + (rank < extra ? 1 : 0) - 1;
  return {start, end};
}

std::vector<int> gatherAndMerge(int rank, int size, const std::vector<int> &local_data) {
  if (rank != 0) {
    int count = static_cast<int>(local_data.size());
    MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    if (count > 0) {
      std::vector<int> send_buffer(local_data);
      MPI_Send(send_buffer.data(), count, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    return {};
  }

  std::vector<std::vector<int>> all_blocks;
  all_blocks.reserve(size);

  all_blocks.emplace_back();
  if (!local_data.empty()) {
    all_blocks.back().assign(local_data.begin(), local_data.end());
  }

  for (int src = 1; src < size; ++src) {
    int count = 0;
    MPI_Recv(&count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    all_blocks.emplace_back();
    if (count > 0) {
      all_blocks.back().resize(count);
      MPI_Recv(all_blocks.back().data(), count, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  std::vector<int> result;
  for (const auto &block : all_blocks) {
    if (block.empty()) {
      continue;
    }
    if (result.empty()) {
      result = block;
    } else {
      std::vector<int> merged;
      merged.reserve(result.size() + block.size());
      std::merge(result.begin(), result.end(), block.begin(), block.end(), std::back_inserter(merged));
      result = std::move(merged);
    }
  }
  return result;
}

}  // namespace

bool BatushinIQuickSortWithSimpleMergeMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &global_input = GetInput();
  int total_size = static_cast<int>(global_input.size());

  if (total_size == 0) {
    GetOutput().clear();
    int dummy = 0;
    MPI_Bcast(&dummy, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return true;
  }

  auto [local_start, local_end] = computeLocalRange(rank, size, total_size);
  int local_count = (local_start <= local_end) ? (local_end - local_start + 1) : 0;

  std::vector<int> local_data;
  if (local_count > 0) {
    local_data.resize(local_count);
  }

  if (rank == 0) {
    if (local_count > 0) {
      std::copy(global_input.begin() + local_start, global_input.begin() + local_start + local_count,
                local_data.begin());
    }
    for (int proc_rank = 1; proc_rank < size; ++proc_rank) {
      auto [s, e] = computeLocalRange(proc_rank, size, total_size);
      int cnt = (s <= e) ? (e - s + 1) : 0;
      if (cnt > 0) {
        std::vector<int> send_buffer(global_input.begin() + s, global_input.begin() + s + cnt);
        MPI_Send(send_buffer.data(), cnt, MPI_INT, proc_rank, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    if (local_count > 0) {
      MPI_Recv(local_data.data(), local_count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  iterativeQuickSort(local_data);

  std::vector<int> result = gatherAndMerge(rank, size, local_data);

  int result_size = 0;
  if (rank == 0) {
    GetOutput() = std::move(result);
    result_size = static_cast<int>(GetOutput().size());
    MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (result_size > 0) {
      MPI_Bcast(GetOutput().data(), result_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (result_size > 0) {
      GetOutput().resize(result_size);
      MPI_Bcast(GetOutput().data(), result_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
  }

  return true;
}

bool BatushinIQuickSortWithSimpleMergeMPI::PostProcessingImpl() {
  return GetOutput().size() == GetInput().size();
}

}  // namespace batushin_i_quick_sort_with_simple_merge
