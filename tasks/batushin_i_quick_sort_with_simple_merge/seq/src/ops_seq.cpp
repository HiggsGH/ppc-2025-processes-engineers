#include "batushin_i_quick_sort_with_simple_merge/seq/include/ops_seq.hpp"

#include <numeric>
#include <stack>
#include <vector>

#include "batushin_i_quick_sort_with_simple_merge/common/include/common.hpp"
#include "util/include/util.hpp"

namespace batushin_i_quick_sort_with_simple_merge {

BatushinIQuickSortWithSimpleMergeSEQ::BatushinIQuickSortWithSimpleMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.size());
}

bool BatushinIQuickSortWithSimpleMergeSEQ::ValidationImpl() {
  return true;
}

bool BatushinIQuickSortWithSimpleMergeSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

namespace {

void IterativeQuickSort(std::vector<int> &data) {
  if (data.size() <= 1) {
    return;
  }
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

    int pivot = data[seg.begin + (seg.end - seg.begin) / 2];
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

}  // namespace

bool BatushinIQuickSortWithSimpleMergeSEQ::RunImpl() {
  IterativeQuickSort(GetOutput());
  return true;
}

bool BatushinIQuickSortWithSimpleMergeSEQ::PostProcessingImpl() {
  return GetOutput().size() == GetInput().size();
}

}  // namespace batushin_i_quick_sort_with_simple_merge
