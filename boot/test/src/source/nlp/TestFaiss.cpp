#include "faiss/IndexFlat.h"
#include "faiss/MetaIndexes.h"
// #include "faiss/IndexIDMap.h"

#include <random>

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

static void testSearch() {
    faiss::IndexFlatL2 db(3);
    faiss::IndexIDMap map(&db);
    // faiss::IndexFlatL2 db(300);
    float a[] {
        1.0, 2.0, 3.0,
        2.0, 2.0, 3.0,
        3.0, 2.0, 3.0,
        1.0, 2.0, 3.0
    };
    faiss::idx_t i[]{1, 2, 3, 4};
    // db.add(4, a);
    map.add_with_ids(4, a, i);
    std::random_device device;
    std::mt19937 rand(device());
    std::uniform_real_distribution<float> urand(0, 1000);
    for(size_t index = 0LL; index < 200'000; ++index) {
        float r[] { urand(rand), urand(rand), urand(rand) };
        map.add_with_ids(1, r, i + 4);
    }
    // SPDLOG_DEBUG("总量：{}", db.ntotal);
    SPDLOG_DEBUG("总量：{}", map.ntotal);
    int k = 3;
    int n = 2;
    float e[] { 1.0, 2.0, 3.0 };
    // float e[] { 0.0, 2.0, 3.0 };
    faiss::idx_t* I = new faiss::idx_t[k * n];
    float* D = new float[k * n];
    // db.search(n, e, k, D, I);
    map.search(n, e, k, D, I);
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