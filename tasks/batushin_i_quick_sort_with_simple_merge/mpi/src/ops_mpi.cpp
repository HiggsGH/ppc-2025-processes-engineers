#include "batushin_i_quick_sort_with_simple_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <stack>
#include <utility>
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

void SortSmallArray(std::vector<int> &data, int begin, int end) {
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

void PartitionArray(std::vector<int> &data, int begin, int end, int &left_end, int &right_begin) {
  if (begin >= end) {
    left_end = begin - 1;
    right_begin = end + 1;
    return;
  }

  int mid = begin + ((end - begin) / 2);
  std::swap(data[mid], data[begin]);
  int pivot = data[begin];

  int i = begin;
  int j = end;
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
  left_end = j;
  right_begin = i;
}

void IterativeQuickSort(std::vector<int> &data) {
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
      SortSmallArray(data, seg.begin, seg.end);
      continue;
    }

    int left_end = 0;
    int right_begin = 0;
    PartitionArray(data, seg.begin, seg.end, left_end, right_begin);

    if (seg.begin <= left_end) {
      stk.push({seg.begin, left_end});
    }
    if (right_begin <= seg.end) {
      stk.push({right_begin, seg.end});
    }
  }
}

std::pair<int, int> ComputeLocalRange(int rank, int size, int total) {
  int base = total / size;
  int extra = total % size;
  int start = (rank * base) + std::min(rank, extra);
  int end = start + base + (rank < extra ? 1 : 0) - 1;
  return std::make_pair(start, end);
}

void DistributeData(int rank, int size, const std::vector<int> &global_input, std::vector<int> &local_data) {
  auto range = ComputeLocalRange(rank, size, static_cast<int>(global_input.size()));
  int local_start = range.first;
  int local_end = range.second;
  int local_count = (local_start <= local_end) ? (local_end - local_start + 1) : 0;

  if (rank == 0) {
    if (local_count > 0) {
      local_data.resize(local_count);
      std::copy(global_input.begin() + local_start, global_input.begin() + local_start + local_count,
                local_data.begin());
    }
    for (int proc_rank = 1; proc_rank < size; ++proc_rank) {
      auto proc_range = ComputeLocalRange(proc_rank, size, static_cast<int>(global_input.size()));
      int s = proc_range.first;
      int e = proc_range.second;
      int cnt = (s <= e) ? (e - s + 1) : 0;
      if (cnt > 0) {
        std::vector<int> send_buffer(global_input.begin() + s, global_input.begin() + s + cnt);
        MPI_Send(send_buffer.data(), cnt, MPI_INT, proc_rank, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    if (local_count > 0) {
      local_data.resize(local_count);
      MPI_Recv(local_data.data(), local_count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

void BroadcastResult(int rank, std::vector<int> &result, std::vector<int> &output) {
  int result_size = 0;

  if (rank == 0) {
    result_size = static_cast<int>(result.size());
    output = std::move(result);
  }
  MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (result_size > 0) {
    if (rank == 0) {
      MPI_Bcast(output.data(), result_size, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
      output.resize(result_size);
      MPI_Bcast(output.data(), result_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
  } else {
    if (rank != 0) {
      output.clear();
    }
  }
}

std::vector<std::vector<int>> CollectAllBlocks(int size, const std::vector<int> &local_data) {
  std::vector<std::vector<int>> all_blocks;
  all_blocks.reserve(size);

  if (!local_data.empty()) {
    all_blocks.emplace_back(local_data.begin(), local_data.end());
  } else {
    all_blocks.emplace_back();
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

  return all_blocks;
}

std::vector<int> MergeTwoBlocks(const std::vector<int> &a, const std::vector<int> &b) {
  std::vector<int> merged;
  merged.reserve(a.size() + b.size());

  auto it1 = a.begin();
  auto it2 = b.begin();
  while (it1 != a.end() && it2 != b.end()) {
    if (*it1 <= *it2) {
      merged.push_back(*it1++);
    } else {
      merged.push_back(*it2++);
    }
  }
  while (it1 != a.end()) {
    merged.push_back(*it1++);
  }
  while (it2 != b.end()) {
    merged.push_back(*it2++);
  }

  return merged;
}

std::vector<int> MergeSortedBlocks(const std::vector<std::vector<int>> &all_blocks) {
  std::vector<int> result;
  for (const auto &block : all_blocks) {
    if (block.empty()) {
      continue;
    }
    if (result.empty()) {
      result.assign(block.begin(), block.end());
    } else {
      result = MergeTwoBlocks(result, block);
    }
  }
  return result;
}

std::vector<int> GatherAndMerge(int rank, int size, const std::vector<int> &local_data) {
  if (rank != 0) {
    int count = static_cast<int>(local_data.size());
    MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    if (count > 0) {
      MPI_Send(local_data.data(), count, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    return {};
  }

  auto all_blocks = CollectAllBlocks(size, local_data);
  return MergeSortedBlocks(all_blocks);
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

  std::vector<int> local_data;
  DistributeData(rank, size, global_input, local_data);

  if (!local_data.empty()) {
    IterativeQuickSort(local_data);
  }

  std::vector<int> result = GatherAndMerge(rank, size, local_data);
  BroadcastResult(rank, result, GetOutput());

  return true;
}

bool BatushinIQuickSortWithSimpleMergeMPI::PostProcessingImpl() {
  return GetOutput().size() == GetInput().size();
}

}  // namespace batushin_i_quick_sort_with_simple_merge
