#include "lifuren/Test.hpp"

#include <set>
#include <mutex>
#include <regex>
#include <thread>
#include <atomic>
#include <fstream>
#include <condition_variable>

#include "nlohmann/json.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Poetry.hpp"
#include "lifuren/String.hpp"
#include "lifuren/EmbeddingClient.hpp"

static bool enable_print_messy_code = false;

// 不能含有空格
static std::wregex word_regex (L"[\\u4e00-\\u9fff]+");   // 词语
static std::wregex title_regex(L"[\\u4e00-\\u9fff・]+"); // 标题

static void print_messy_code(const std::string& file, const std::string& title, const std::string& value, const std::wregex& regex) {
    if(!enable_print_messy_code || value.empty()) {
        return;
    }
    if(!std::regex_match(lifuren::string::to_wstring(value), regex)) {
        SPDLOG_INFO("文本乱码：{} {} {}", file, title, value);
    }
}

static void print(const char* title, const std::map<std::string, int64_t>& map) {
    std::vector<std::pair<std::string, int64_t>> order(map.begin(), map.end());
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& z) {
        return a.second > z.second;
    });
    for(const auto& [k, v] : order) {
        SPDLOG_DEBUG("{}：{} - {}", title, k, v);
    }
}

[[maybe_unused]] static void testPepper() {
    std::vector<std::string> files;
    // lifuren::file::listFile(files, lifuren::config::CONFIG.mark.begin()->path, { ".json" });
    lifuren::file::listFile(files, lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren", "mark", "ci" }).string(), { ".json" });
    // lifuren::file::listFile(files, lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren", "mark", "shi" }).string(), { ".json" });
    // lifuren::file::listFile(files, lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren", "mark", "songshi" }).string(), { ".json" });
    // lifuren::file::listFile(files, lifuren::file::join({ lifuren::config::CONFIG.tmp, "lifuren", "mark", "tangshi" }).string(), { ".json" });
    int64_t fSize    = 0LL;
    int64_t wSize    = 0LL;
    int64_t count    = 0LL;
    int64_t total    = 0LL;
    int64_t ciCount  = 0LL;
    int64_t ciTotal  = 0LL;
    int64_t shiCount = 0LL;
    int64_t shiTotal = 0LL;
    std::set<std::string> words;
    std::map<std::string, int64_t> unciCount;
    std::map<std::string, int64_t> unshiCount;
    std::map<std::string, int64_t> rhythmCount;
    for(const auto& file : files) {
        ++fSize;
        std::string json = std::move(lifuren::file::loadFile(file));
        auto poetries = nlohmann::json::parse(json);
        for(const auto& poetry : poetries) {
            const bool ci = poetry.contains("rhythmic");
            lifuren::poetry::Poetry value = poetry;
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
                // SPDLOG_DEBUG("匹配成功：{} - {}", value.rhythmPtr->rhythm, value.simpleSegment);
                ++iter->second;
                ++count;
                if(ci) {
                    ++ciCount;
                } else {
                    ++shiCount;
                }
                value.participle();
                print_messy_code(file, value.title, value.title,    title_regex);
                print_messy_code(file, value.title, value.author,   word_regex);
                print_messy_code(file, value.title, value.rhythmic, title_regex);
                for(const auto& word : value.participleParagraphs) {
                    ++wSize;
                    words.insert(word);
                    print_messy_code(file, value.title, word, word_regex);
                }
            } else {
                if(ci) {
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
                // SPDLOG_DEBUG("匹配失败：{}", poetry.dump());
                // return;
            }
            if(total % 1000 == 0) {
                SPDLOG_DEBUG("当前数量：{} / {} / {}", fSize, count, total);
            }
        }
    }
    print("匹配格律", rhythmCount);
    // print("未知词格律", unciCount);
    // print("未知诗格律", unshiCount);
    SPDLOG_DEBUG("累计处理文件数量：{}", fSize);
    SPDLOG_DEBUG("诗词匹配格律数量：{} / {}", count,    total);
    SPDLOG_DEBUG("词匹配格律的数量：{} / {}", ciCount,  ciTotal);
    SPDLOG_DEBUG("诗匹配格律的数量：{} / {}", shiCount, shiTotal);
    SPDLOG_DEBUG("格律累计分词数量：{} / {}", words.size(), wSize);
    auto embeddingClient = lifuren::EmbeddingClient::getClient("ollama");
    std::ofstream output;
    output.open(lifuren::file::join({ lifuren::config::CONFIG.tmp, "pepper", "pepper.word" }).string(), std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    if(!output.is_open()) {
        output.close();
        SPDLOG_WARN("文件打开失败");
        return;
    }
    std::mutex mutex;
    std::condition_variable condition;
    std::vector<std::string> vector;
    vector.reserve(words.size());
    vector.assign(words.begin(), words.end());
    const int batch = 10;
    std::atomic_int countDown(batch);
    const int batchSize = vector.size() / batch;
    for(int i = 0; i < batch; ++i) {
        std::thread thread([i, &mutex, &output, &vector, &countDown, &batchSize, &condition, &embeddingClient]() {
            int index = 0;
            SPDLOG_DEBUG("启动线程：{} {} {}", i , batch - 1, i == (batch - 1));
            auto beg = vector.begin() + (i * batchSize);
            auto end = (i == batch - 1) ? vector.end() : beg + batchSize;
            for(; beg != end; ++beg) {
                auto x = std::move(embeddingClient->getVector(*beg));
                SPDLOG_DEBUG("处理词语：{} {} {}", *beg, beg->size(), x.size());
                std::lock_guard<std::mutex> lock(mutex);
                size_t iSize = beg->size();
                output.write(reinterpret_cast<char*>(&iSize), sizeof(size_t));
                output.write(beg->data(), beg->size());
                size_t xSize = x.size();
                output.write(reinterpret_cast<char*>(&xSize), sizeof(size_t));
                output.write(reinterpret_cast<char*>(x.data()), xSize * sizeof(float));
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
    output.flush();
    output.close();
}

LFR_TEST(
    testPepper();
);
