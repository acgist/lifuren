#include "faiss/IndexFlat.h"

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

using idx_t = int64_t;

static void testSearch() {
    faiss::IndexFlatL2 db(3);
    // faiss::IndexFlatL2 db(300);
    float a[]{1.0, 2.0, 3.0};
    idx_t ai[]{1};
    float b[]{2.0, 2.0, 3.0};
    idx_t bi[]{2};
    float c[]{3.0, 2.0, 3.0};
    idx_t ci[]{3};
    float d[]{4.0, 2.0, 3.0};
    idx_t di[]{4};
    db.add(1, a);
    db.add(1, b);
    db.add(1, c);
    db.add(1, d);
    // db.add_with_ids(1, a, ai);
    // db.add_with_ids(1, b, bi);
    // db.add_with_ids(1, c, ci);
    // db.add_with_ids(1, d, di);
    int k = 3;
    int n = 2;
    float e[]{1.0, 2.0, 3.0};
    // float e[]{0.0, 2.0, 3.0};
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
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testSearch();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}