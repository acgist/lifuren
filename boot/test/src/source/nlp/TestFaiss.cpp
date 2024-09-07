#include "faiss/IndexFlat.h"

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

using idx_t = int64_t;

static void testSearch() {
    faiss::IndexFlatL2 db(3);
    // faiss::IndexFlatL2 db(300);
    float a[] {
        1.0, 2.0, 3.0,
        2.0, 2.0, 3.0,
        3.0, 2.0, 3.0,
        1.0, 2.0, 3.0
    };
    idx_t i[]{1, 2, 3, 4};
    db.add(4, a);
    SPDLOG_DEBUG("总量：{}", db.ntotal);
    int k = 3;
    int n = 2;
    float e[] { 1.0, 2.0, 3.0 };
    // float e[] { 0.0, 2.0, 3.0 };
    idx_t* I = new idx_t[k * n];
    float* D = new float[k * n];
    db.search(n, e, k, D, I);
    std::string is = "\n";
    std::string ds = "\n";
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < k; ++j) {
            is += " " + std::to_string(I[i * k + j]);
            ds += " " + std::to_string(D[i * k + j]);
        }
        is += "\n";
        ds += "\n";
    }
    SPDLOG_DEBUG("I = {}", is);
    SPDLOG_DEBUG("D = {}", ds);
    delete[] I;
    delete[] D;
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testSearch();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}