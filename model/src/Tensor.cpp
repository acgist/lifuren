#include "lifuren/Tensor.hpp"

#include "spdlog/spdlog.h"

std::string lifuren::tensor::print(const ggml_tensor* tensor, const bool log) {
    std::string message;
    const int64_t& a = tensor->ne[0];
    const int64_t& b = tensor->ne[1];
    const int64_t& c = tensor->ne[2];
    const int64_t& d = tensor->ne[3];
    message += fmt::format(R"(
dims         = {} * {} * {} * {}
nbytes       = {}
nelements    = {}
element_size = {}
)", d, c, b, a, ggml_nbytes(tensor), ggml_nelements(tensor), ggml_element_size(tensor));
    const float* data = ggml_get_data_f32(tensor);
    for(int64_t i = 0LL, di = 0LL; di < d; ++di) {
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
        SPDLOG_DEBUG("\n{}", message);
    }
    return message;
}
