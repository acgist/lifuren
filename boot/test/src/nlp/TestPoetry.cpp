#include "lifuren/Test.hpp"

#include <map>
#include <set>
#include <regex>

#include "nlohmann/json.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/String.hpp"
#include "lifuren/RAGClient.hpp"
#include "lifuren/EmbeddingClient.hpp"
#include "lifuren/poetry/Poetry.hpp"
#include "lifuren/poetry/PoetryDataset.hpp"

static bool enable_print_messy_code = false;

// 不能含有空格
static std::wregex word_regex (L"[\\u4e00-\\u9fff]{1,8}"); // 词语
static std::wregex title_regex(L"[\\u4e00-\\u9fff・]+");   // 标题

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

[[maybe_unused]] static void testPoetry() {
    nlohmann::json json = nlohmann::json::parse(R"(
    {
        "author": "杜甫",
        "paragraphs": [
            "国破山河在，城春草木深。",
            "感时花溅泪，恨别鸟惊心。",
            "烽火连三月，家书抵万金。",
            "白头掻更短，浑欲不胜簪。"
        ],
        "rhythmic": "春望"
    }
    )");
    // nlohmann::json json = nlohmann::json::parse(R"(
    // {
    //     "author": "欧阳修",
    //     "paragraphs": [
    //         "庭院深深深几许，杨柳堆烟，帘幕无重数。",
    //         "玉勒雕鞍游冶处，楼高不见章台路。",
    //         "雨横风狂三月暮，门掩黄昏，无计留春住。",
    //         "泪眼问花花不语，乱红飞过秋千去。"
    //     ],
    //     "rhythmic": "蝶恋花"
    // }
    // )");
    // nlohmann::json json = nlohmann::json::parse(R"(
    // {
    //     "author": "朱敦儒",
    //     "paragraphs": [
    //         "我是清都山水郎，天教懒慢带疏狂。",
    //         "曾批给露支风敕，累奏留云借月章。",
    //         "诗万首，酒千觞，几曾着眼看侯王？玉楼金阙慵归去，且插梅花醉洛阳。"
    //     ],
    //     "rhythmic": "鹧鸪天"
    // }
    // )");
    lifuren::poetry::Poetry poetry = json;
    poetry.preproccess();
    SPDLOG_DEBUG("匹配格律：\n{}", poetry.matchRhythm());
    SPDLOG_DEBUG("原始段落：\n{}", poetry.segment);
    SPDLOG_DEBUG("朴素段落：\n{}", poetry.simpleSegment);
    if(poetry.matchRhythm()) {
        poetry.participle();
        SPDLOG_DEBUG("分词段落：\n{}", poetry.participleSegment);
    }
    lifuren::poetry::Poetry diff = json;
    assert(diff == poetry);
}

[[maybe_unused]] static void testDataset() {
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
}

[[maybe_unused]] static void testRAGEmbedding() {
    const std::string rag       = "faiss";
    const std::string path      = lifuren::file::join({ lifuren::config::CONFIG.tmp, "poetry-embedding" }).string();
    const std::string embedding = "pepper";
    std::shared_ptr<lifuren::RAGClient> client = std::move(lifuren::RAGClient::getClient(rag, path, embedding));
    auto embeddingFunction = std::bind(&lifuren::poetry::ragEmbedding, client, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
    lifuren::dataset::allDatasetPreprocessing(path, lifuren::config::EMBEDDING_MODEL_FILE, embeddingFunction);
}

[[maybe_unused]] static void testPepperEmbedding() {
    lifuren::dataset::allDatasetPreprocessing(
        lifuren::file::join({ lifuren::config::CONFIG.tmp, "poetry-embedding" }).string(),
        lifuren::config::PEPPER_MODEL_FILE,
        &lifuren::poetry::pepperEmbedding,
        true
    );
}

[[maybe_unused]] static void testOllamaEmbeddingClient() {
    lifuren::OllamaEmbeddingClient client{ lifuren::config::CONFIG.tmp };
    const auto v = std::move(client.getVector("李夫人"));
    SPDLOG_DEBUG("v length = {}", v.size());
}

[[maybe_unused]] static void testPepperEmbeddingClient() {
    // lifuren::PepperEmbeddingClient ref{};
    {
        lifuren::PepperEmbeddingClient client{ lifuren::config::CONFIG.tmp };
        auto v = std::move(client.getVector("东风"));
        // auto v = std::move(client.getVector({ "李", "夫", "人"}));
        SPDLOG_DEBUG("v length = {}", v.size());
    }
    SPDLOG_DEBUG("释放1");
    SPDLOG_DEBUG("释放2");
    SPDLOG_DEBUG("释放3");
}

[[maybe_unused]] static void testRAGClientIndex() {
    lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    // lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "pepper" };
    // lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    // lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "pepper" };
    client.index("猪");
    client.index("牛");
    client.index("马");
    client.index("马");
    client.index("马");
    client.index("桃子");
    client.index("桃子");
    client.index("桃子");
    client.index("苹果");
    client.index("李子");
}

[[maybe_unused]] static void testRAGClientSearch() {
    lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    // lifuren::FaissRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "pepper" };
    // lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "ollama" };
    // lifuren::ElasticSearchRAGClient client{ lifuren::file::join({lifuren::config::CONFIG.tmp, "docs"}).string(), "pepper" };
    auto a = client.search("狗");
    for(const auto& v : a) {
        SPDLOG_DEBUG("狗 = {}", v);
    }
    auto b = client.search("水果");
    // auto b = client.search("草莓");
    for(const auto& v : b) {
        SPDLOG_DEBUG("水果 = {}", v);
        // SPDLOG_DEBUG("草莓 = {}", v);
    }
}

[[maybe_unused]] static void testLoadPoetryFileDataset() {
    auto loader = lifuren::poetry::loadFileDatasetLoader(5, lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "embedding.model"}).string());
    lifuren::logTensor("诗词特征", loader->begin()->data.sizes());
    lifuren::logTensor("诗词标签", loader->begin()->target.sizes());
}

LFR_TEST(
    // testPoetry();
    // testDataset();
    testRAGEmbedding();
    // testPepperEmbedding();
    // testOllamaEmbeddingClient();
    // testPepperEmbeddingClient();
    // testRAGClientIndex();
    // testRAGClientSearch();
    // testLoadPoetryFileDataset();
);
