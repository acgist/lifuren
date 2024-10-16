#include "lifuren/Tensor.hpp"

#include "spdlog/spdlog.h"

std::string lifuren::tensor::print(const ggml_tensor* tensor, const bool log) {
    std::string message;
    if(tensor == nullptr) {
        return message;
    }
    const int64_t& a = tensor->ne[0];
    const int64_t& b = tensor->ne[1];
    const int64_t& c = tensor->ne[2];
    const int64_t& d = tensor->ne[3];
    message += fmt::format(R"(
dims         = {} * {} * {} * {}
nbytes       = {}
nelements    = {}
element_size = {}

)", a, b, c, d, ggml_nbytes(tensor), ggml_nelements(tensor), ggml_element_size(tensor));
    const float* data = ggml_get_data_f32(tensor);
    if(d == 1LL && c == 1L && b == 1L) {
        for(int64_t ai = 0LL; ai < a; ++ai) {
            message += fmt::format(" {: >10.6f}\n", data[ai]);
        }
    } else if(d == 1LL && c == 1L) {
        for(int64_t ai = 0LL; ai < a; ++ai) {
            message += " ";
            for(int64_t bi = 0LL; bi < b; ++bi) {
                message += fmt::format("{: >10.6f} ", data[
                    b * ai +
                        bi
                ]);
            }
            message += '\n';
        }
    } else if(d == 1LL) {
        for(int64_t ai = 0LL; ai < a; ++ai) {
            message += "{\n";
                for(int64_t bi = 0LL; bi < b; ++bi) {
                    message += " ";
                    for(int64_t ci = 0LL; ci < c; ++ci) {
                        message += fmt::format("{: >10.6f} ", data[
                                b * c * ai +
                                    c * bi +
                                        ci
                        ]);
                    }
                    message += '\n';
                }
            message += fmt::format("{} ({},.,.)\n", "}", ai);
        }
    } else {
        for(int64_t bi = 0LL; bi < b; ++bi) {
            message += "{\n";
            for(int64_t ai = 0LL; ai < a; ++ai) {
                message += " [\n";
                for(int64_t ci = 0LL; ci < c; ++ci) {
                    message += "  ";
                    for(int64_t di = 0LL; di < d; ++di) {
                        message += fmt::format("{: >10.6f} ", data[
                            b * c * d * ai +
                                c * d * bi +
                                    d * ci +
                                        di
                        ]);
                    }
                    message += '\n';
                }
                message += fmt::format(" ] ({},{},.,.)\n", ai, bi);
            }
            message += "}\n";
        }
    }
    if(log) {
        SPDLOG_DEBUG("\n{}", message);
    }
    return message;
}
