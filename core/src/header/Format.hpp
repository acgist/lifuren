/**
 * 日志格式输出
 */
#pragma once

#include "Logger.hpp"

#include "torch/torch.h"

#include "spdlog/fmt/ostr.h"

/**
 * Tensor格式化
 */
template <> struct fmt::formatter<at::Tensor> : ostream_formatter {};
    
namespace lifuren {

}
