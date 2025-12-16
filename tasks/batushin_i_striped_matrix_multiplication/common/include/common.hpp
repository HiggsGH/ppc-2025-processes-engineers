#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace batushin_i_striped_matrix_multiplication {

using InType = std::tuple<size_t, size_t, std::vector<double>, size_t, size_t, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<std::string, InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace batushin_i_striped_matrix_multiplication
