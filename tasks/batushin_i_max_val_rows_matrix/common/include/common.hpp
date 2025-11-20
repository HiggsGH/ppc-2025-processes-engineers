#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace batushin_i_max_val_rows_matrix {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace batushin_i_max_val_rows_matrix
