#include "lifuren/Test.hpp"

#include <random>

#include "faiss/index_io.h"
#include "faiss/IndexFlat.h"
#include "faiss/MetaIndexes.h"

#include "lifuren/Files.hpp"

[[maybe_unused]] static void testSearch() {
    faiss::IndexFlatL2 db(3);
    faiss::IndexIDMap  map(&db);
    float a[] {
        1.0, 2.0, 3.0,
        2.0, 2.0, 3.0,
        3.0, 2.0, 3.0,
        1.0, 2.0, 3.0
    };
    int64_t i[] { 1, 2, 3, 4 };
    // db.add(4, a);
    map.add_with_ids(4, a, i);
    std::random_device device;
    std::mt19937 rand(device());
    std::uniform_real_distribution<float> urand(0, 1000);
    for(size_t index = 0LL; index < 200'000; ++index) {
        float r[] { urand(rand), urand(rand), urand(rand) };
        map.add_with_ids(1, r, i + 4);
    }
    SPDLOG_DEBUG("总量：{}", db.ntotal);
    SPDLOG_DEBUG("总量：{}", map.ntotal);
    int k = 3;
    int n = 1;
    float e[] { 1.0, 2.0, 3.0 };
    // float e[] { 0.0, 2.0, 3.0 };
    float  * D = new float  [k * n];
    int64_t* I = new int64_t[k * n];
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
    I = nullptr;
    D = nullptr;
}

[[maybe_unused]] static void testIO() {
    faiss::IndexFlatL2 db(3);
    faiss::IndexIDMap  map(&db);
    float a[] {
        1.0, 2.0, 3.0,
        2.0, 2.0, 3.0,
        3.0, 2.0, 3.0,
        1.0, 2.0, 3.0
    };
    int64_t i[] { 1, 2, 3, 4 };
    map.add_with_ids(4, a, i);
    const std::string file = lifuren::files::join({lifuren::config::CONFIG.tmp, "faiss.db"}).string();
    faiss::write_index(&map, file.c_str());
    auto load = faiss::read_index(file.c_str());
    assert(load->ntotal == db.ntotal);
    assert(load->ntotal == map.ntotal);
    float xx[] { 1.0, 2.0, 3.0 };
    float  * dd = new float  [1 * 2];
    int64_t* ll = new int64_t[1 * 2];
    load->search(1, xx, 2, dd, ll);
    std::copy(dd, dd + 2, std::ostream_iterator<float>(std::cout, " "));
    std::cout << '\n';
    std::copy(ll, ll + 6, std::ostream_iterator<int64_t>(std::cout, " "));
    std::cout << '\n';
    delete[] dd;
    delete[] ll;
    dd = nullptr;
    ll = nullptr;
}

LFR_TEST(
    testIO();
    // testSearch();
);
