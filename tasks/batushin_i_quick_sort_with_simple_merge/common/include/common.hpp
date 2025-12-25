#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace batushin_i_quick_sort_with_simple_merge {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::tuple<std::string, InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace batushin_i_quick_sort_with_simple_merge
