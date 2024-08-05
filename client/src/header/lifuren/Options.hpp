#include <string>

// cv

struct ImageOptions {
};

struct VideoOptions {
};

// nlp
struct NlpOptions {

    // 平台
    std::string platform;
    // 模型
    std::string model;

};

struct EmbeddingOptions {

    // 平台
    std::string platform;
    // 模型
    std::string model;

};

struct SearchOptions {

};
