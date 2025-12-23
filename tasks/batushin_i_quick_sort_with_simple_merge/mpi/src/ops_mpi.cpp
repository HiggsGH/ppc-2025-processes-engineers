#include "batushin_i_quick_sort_with_simple_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <stack>
#include <vector>

#include "batushin_i_quick_sort_with_simple_merge/common/include/common.hpp"
#include "util/include/util.hpp"

namespace batushin_i_quick_sort_with_simple_merge {

BatushinIQuickSortWithSimpleMergeMPI::BatushinIQuickSortWithSimpleMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool BatushinIQuickSortWithSimpleMergeMPI::ValidationImpl() {
  int initialized;
  MPI_Initialized(&initialized);
  return initialized != 0;
}

bool BatushinIQuickSortWithSimpleMergeMPI::PreProcessingImpl() {
  return true;
}

namespace {

void IterativeQuickSort(std::vector<int> &data) {
  if (data.size() <= 1) {
    return;
  }

  const int THRESHOLD = 16;

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

    if (seg.end - seg.begin + 1 <= THRESHOLD) {
      for (int i = seg.begin + 1; i <= seg.end; ++i) {
        int key = data[i];
        int j = i - 1;
        while (j >= seg.begin && data[j] > key) {
          data[j + 1] = data[j];
          --j;
        }
        data[j + 1] = key;
      }
      continue;
    }

    int mid = seg.begin + (seg.end - seg.begin) / 2;
    std::swap(data[mid], data[seg.begin]);
    int pivot = data[seg.begin];

    int i = seg.begin - 1;
    int j = seg.end + 1;
    while (true) {
      do {
        i++;
      } while (data[i] < pivot);
      do {
        j--;
      } while (data[j] > pivot);
      if (i >= j) {
        break;
      }
      std::swap(data[i], data[j]);
    }

    stk.push({seg.begin, j});
    stk.push({j + 1, seg.end});
  }
}

std::pair<int, int> ComputeLocalRange(int rank, int size, int total) {
  int base = total / size;
  int extra = total % size;
  int start = rank * base + std::min(rank, extra);
  int end = start + base + (rank < extra ? 1 : 0) - 1;
  return {start, end};
}

std::vector<int> ParallelMergeTree(int rank, int size, std::vector<int> &local_data) {
  int steps = 0;
  int temp = size;
  while (temp > 1) {
    steps++;
    temp >>= 1;
  }

  std::vector<int> current = std::move(local_data);

  for (int step = 0; step < steps; ++step) {
    int partner = rank ^ (1 << step);

    if (partner >= size) {
      continue;
    }

    if (rank < partner) {
      int partner_size;
      MPI_Recv(&partner_size, 1, MPI_INT, partner, step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if (partner_size == 0) {
        continue;
      }

      std::vector<int> partner_data(partner_size);
      MPI_Recv(partner_data.data(), partner_size, MPI_INT, partner, step, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<int> merged;
      merged.reserve(current.size() + partner_data.size());
      std::merge(current.begin(), current.end(), partner_data.begin(), partner_data.end(), std::back_inserter(merged));
      current = std::move(merged);

    } else {
      int current_size = static_cast<int>(current.size());
      MPI_Send(&current_size, 1, MPI_INT, partner, step, MPI_COMM_WORLD);
      if (current_size > 0) {
        MPI_Send(current.data(), current_size, MPI_INT, partner, step, MPI_COMM_WORLD);
      }
      return {};
    }
  }

  return current;
}

}  // namespace

bool BatushinIQuickSortWithSimpleMergeMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &global_input = GetInput();
  int total_size = static_cast<int>(global_input.size());

  auto [local_start, local_end] = ComputeLocalRange(rank, size, total_size);
  int local_count = (local_start <= local_end) ? (local_end - local_start + 1) : 0;

  std::vector<int> local_data(local_count, 0);

  if (rank == 0) {
    if (local_count > 0) {
      std::copy(global_input.begin() + local_start, global_input.begin() + local_start + local_count,
                local_data.begin());
    }
    for (int r = 1; r < size; ++r) {
      auto [s, e] = ComputeLocalRange(r, size, total_size);
      int cnt = (s <= e) ? (e - s + 1) : 0;
      if (cnt > 0) {
        MPI_Send(const_cast<int *>(global_input.data() + s), cnt, MPI_INT, r, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    if (local_count > 0) {
      MPI_Recv(local_data.data(), local_count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  IterativeQuickSort(local_data);

  std::vector<int> result = ParallelMergeTree(rank, size, local_data);

  if (!result.empty()) {
    GetOutput() = std::move(result);

    int result_size = static_cast<int>(GetOutput().size());
    MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (result_size > 0) {
      MPI_Bcast(GetOutput().data(), result_size, MPI_INT, 0, MPI_COMM_WORLD);
    }

  } else {
    int result_size;
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
