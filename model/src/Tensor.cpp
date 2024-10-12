#include "lifuren/Tensor.hpp"

#include "spdlog/spdlog.h"

std::string lifuren::tensor::print(const ggml_tensor* tensor, const bool log) {
    int64_t i = 0;
    std::string message = "size = ";
    const float* data = ggml_get_data_f32(tensor);
    const int64_t& a = tensor->ne[0];
    message += std::to_string(a);
    message += " * ";
    const int64_t& b = tensor->ne[1];
    message += std::to_string(b);
    message += " * ";
    const int64_t& c = tensor->ne[2];
    message += std::to_string(c);
    message += " * ";
    const int64_t& d = tensor->ne[3];
    message += std::to_string(d);
    message += "\n";
    for(int64_t di = 0LL; di < d; ++di) {
        message += "{\n";
        for(int64_t ci = 0LL; ci < c; ++ci) {
            message += " [\n";
            for(int64_t bi = 0LL; bi < b; ++bi) {
                message += "  ";
                for(int64_t ai = 0LL; ai < a; ++ai, ++i) {
                    message += fmt::format("{: >10.6f} ", data[i]);
                }
                message += '\n';
            }
            message += " ]\n";
        }
        message += "}\n";
    }
    if(log) {
        SPDLOG_DEBUG("\n\n{}", message);
    }
    return message;
}
