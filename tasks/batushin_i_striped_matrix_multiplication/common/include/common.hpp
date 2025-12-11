#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace batushin_i_striped_matrix_multiplication {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace batushin_i_striped_matrix_multiplication
