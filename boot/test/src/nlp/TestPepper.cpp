#include "lifuren/Test.hpp"

#include <set>
#include <mutex>
#include <thread>
#include <atomic>
#include <fstream>
#include <condition_variable>

#include "nlohmann/json.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Poetry.hpp"
#include "lifuren/EmbeddingClient.hpp"

[[maybe_unused]] static void testPepper() {
    std::vector<std::string> files;
    lifuren::file::listFile(files, lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren", "mark", "ci" }).string(), { ".json" });
    // lifuren::file::listFile(files, lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren", "mark", "shi" }).string(), { ".json" });
    // lifuren::file::listFile(files, lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren", "mark", "songshi" }).string(), { ".json" });
    // lifuren::file::listFile(files, lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren", "mark", "tangshi" }).string(), { ".json" });
    // lifuren::file::listFile(files, lifuren::config::CONFIG.mark.begin()->path, { ".json" });
    int64_t fSize    = 0LL;
    int64_t wSize    = 0LL;
    int64_t count    = 0LL;
    int64_t total    = 0LL;
    int64_t ciCount  = 0LL;
    int64_t ciTotal  = 0LL;
    int64_t shiCount = 0LL;
    int64_t shiTotal = 0LL;
    std::map<std::string, int64_t> unciCount;
    std::map<std::string, int64_t> unshiCount;
    std::map<std::string, int64_t> rhythmCount;
    std::set<std::string> words;
    for(const auto& file : files) {
        ++fSize;
        std::string&& json = lifuren::file::loadFile(file);
        auto&& poetries = nlohmann::json::parse(json);
        for(const auto& poetry : poetries) {
            lifuren::poetry::Poetry value = poetry;
            const bool ci = poetry.contains("rhythmic");
            value.preproccess();
            ++total;
            if(ci) {
                ++ciTotal;
            } else {
                ++shiTotal;
            }
            if(value.matchRhythm()) {
                auto iter = rhythmCount.find(value.rhythmPtr->rhythm);
                if(iter == rhythmCount.end()) {
                    iter = rhythmCount.emplace(value.rhythmPtr->rhythm, 0).first;
                }
                ++iter->second;
                // SPDLOG_DEBUG("匹配成功：{} - {}", value.rhythmPtr->rhythm, value.simpleSegment);
                ++count;
                if(ci) {
                    ++ciCount;
                } else {
                    ++shiCount;
                }
                value.participle();
                for(const auto& word : value.participleParagraphs) {
                    words.insert(word);
                    ++wSize;
                }
            } else {
                if(poetry.contains("rhythmic")) {
                    auto rhythm = poetry.at("rhythmic").get<std::string>();
                    auto iter = unciCount.find(rhythm);
                    if(iter == unciCount.end()) {
                        iter = unciCount.emplace(rhythm, 0).first;
                    }
                    --iter->second;
                } else {
                    auto rhythm = poetry.at("title").get<std::string>();
                    auto iter = unshiCount.find(rhythm);
                    if(iter == unshiCount.end()) {
                        iter = unshiCount.emplace(rhythm, 0).first;
                    }
                    --iter->second;
                }
                // SPDLOG_DEBUG("匹配失败：{}", rhythm);
                // SPDLOG_DEBUG("匹配失败：{}", poetry.dump());
                // return;
            }
            if(total % 1000 == 0) {
                SPDLOG_DEBUG("当前数量：{} / {} / {}", fSize, count, total);
            }
        }
    }
    std::vector<std::pair<std::string, int64_t>> order(rhythmCount.begin(), rhythmCount.end());
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& z) {
        return a.second > z.second;
    });
    for(const auto& [k, v] : order) {
        SPDLOG_DEBUG("匹配格律：{} - {}", k, v);
    }
    // order.clear();
    // order.insert(order.begin(), unciCount.begin(), unciCount.end());
    // std::sort(order.begin(), order.end(), [](const auto& a, const auto& z) {
    //     return a.second > z.second;
    // });
    // for(const auto& [k, v] : order) {
    //     SPDLOG_DEBUG("未知词格律：{} - {}", k, v);
    // }
    // order.clear();
    // order.insert(order.begin(), unshiCount.begin(), unshiCount.end());
    // std::sort(order.begin(), order.end(), [](const auto& a, const auto& z) {
    //     return a.second > z.second;
    // });
    // for(const auto& [k, v] : order) {
    //     SPDLOG_DEBUG("未知诗格律：{} - {}", k, v);
    // }
    SPDLOG_DEBUG("诗词总数：{} / {} / {}", fSize, count, total);
    SPDLOG_DEBUG("匹配词总数：{} / {}", ciCount,  ciTotal);
    SPDLOG_DEBUG("匹配诗总数：{} / {}", shiCount, shiTotal);
    SPDLOG_DEBUG("累计词数：{} / {}", words.size(), wSize);
    auto embeddingClient = lifuren::EmbeddingClient::getClient("ollama");
    std::ofstream output;
    output.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "pepper", "pepper.word" }).string());
    std::mutex mutex;
    std::condition_variable condition;
    std::vector<std::string> vector;
    vector.reserve(words.size());
    vector.assign(words.begin(), words.end());
    const int batch = 10;
    std::atomic_int countDown(batch);
    const int batchSize = vector.size() / batch;
    for(int i = 0; i < 10; ++i) {
        auto beg = vector.begin() + i * batchSize;
        auto end = (i == batch - 1) ? vector.end() : vector.begin() + ((i + 1) * batchSize);
        std::thread thread([&]() {
            int index = 0;
            for(; beg != end; ++beg) {
                auto x = std::move(embeddingClient->getVector(*beg));
                std::lock_guard<std::mutex> lock(mutex);
                size_t iSize = beg->size();
                output.write(reinterpret_cast<char*>(&iSize), sizeof(size_t));
                output.write(beg->data(), beg->size());
                size_t xSize = x.size();
                output.write(reinterpret_cast<char*>(&xSize), sizeof(size_t));
                output.write(reinterpret_cast<char*>(x.data()), x.size() * sizeof(float));
                if(++index % 100 == 0) {
                    SPDLOG_DEBUG("处理数量：{} - {}", i, index);
                }
            }
            std::lock_guard<std::mutex> lock(mutex);
            --countDown;
            condition.notify_all();
        });
        thread.detach();
    }
    std::unique_lock<std::mutex> lock(mutex);
    while(countDown != 0) {
        condition.wait(lock);
    }
    SPDLOG_INFO("完成");
}

LFR_TEST(
    testPepper();
);
