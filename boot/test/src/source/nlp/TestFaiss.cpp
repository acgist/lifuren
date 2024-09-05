#include "faiss/IndexFlat.h"

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

using idx_t = int64_t;

static void testSearch() {
    faiss::IndexFlatL2 db(3);
    float a[]{1.0, 2.0, 3.0};
    db.add(1, a);
    int k = 5;
    idx_t* I = new idx_t[k * 5];
    float* D = new float[k * 5];
    db.search(5, a, k, D, I);
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testSearch();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}