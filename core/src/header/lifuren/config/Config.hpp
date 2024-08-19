/**
 * 系统设置
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_CONFIG_CONFIG_HPP
#define LFR_HEADER_CORE_CONFIG_CONFIG_HPP

#include <set>
#include <map>
#include <list>
#include <string>

namespace lifuren::config {

const char* const CONFIG_PATH = "../config/config.yml";

// 前置声明
class Config;

extern std::string httpServerHost;
extern int         httpServerPort;
extern const std::string CONFIG_CHAT;
extern const std::string CONFIG_IMAGE_MARK;
extern const std::string CONFIG_POETRY_MARK;
extern const std::string CONFIG_DOCUMENT_MARK;
extern const std::string CONFIG_OPENAI;
extern const std::string CONFIG_OLLAMA;
extern const std::string CONFIG_ELASTICSEARCH;
extern const std::string CONFIG_HTTP_SERVER_HOST;
extern const std::string CONFIG_HTTP_SERVER_PORT;

extern lifuren::config::Config CONFIG;

/**
 * 损失函数
 */
enum class Loss {

    NONE               = 0, // 没有损失函数
    MSE_LOSS           = 1, // 均方误差损失函数
    NLL_LOSS           = 2, // 负对数似然损失函数
    CROSS_ENTROPY_LOSS = 3, // 交叉熵损失函数

};

/**
 * 激活函数
 */
enum class Activation {

    NONE    = 0, // 没有激活函数
    RELU    = 1, // ReLU
    TANH    = 2, // 双曲正切函数
    SIGMOID = 3, // Sigmoid
    SOFTMAX = 4, // Softmax

};

/**
 * 正则函数
 */
enum class Regularization {

    NONE       = 0, // 没有正则
    L1         = 1, // L1
    L2         = 2, // L2
    DROPOUT    = 3, // Dropout
    BATCH_NORM = 4, // BatchNorm

};

/**
 * 聊天页面配置
 */
struct ChatConfig {

    // RAG文档资料数量
    int ragSize = 1;
    // 终端名称
    std::string client;
    // 终端列表
    std::set<std::string> clients;

};

/**
 * 图片生成页面配置
 */
struct ImageConfig {

};

/**
 * 诗词生成页面配置
 */
struct PoetryConfig {

};

/**
 * REST配置
 */
struct RestConfig {

    // 接口地址
    std::string api;
    // 账号
    std::string username;
    // 密码
    std::string password;
    // 授权方式
    std::string authType;
    // 授权地址
    std::string authPath;

};

/**
 * LLM配置
 */
struct LLMConfig {

    double topP;
    size_t topK;
    double temperature;
    std::map<std::string, std::string> options{};

};

/**
 * 聊天终端
 */
struct ChatClientConfig : LLMConfig {

    std::string path;
    std::string model;

};

/**
 * 词嵌入终端
 */
struct EmbeddingClientConfig {

    std::string path;
    std::string model;
    std::map<std::string, std::string> options{};

};

/**
 * OpenAi配置
 */
struct OpenAiConfig : RestConfig {

    // 聊天终端
    ChatClientConfig chatClient;
    // 词嵌入终端
    EmbeddingClientConfig embeddingClient;

};

/**
 * Ollama配置
 */
struct OllamaConfig : RestConfig {

    // 聊天终端
    ChatClientConfig chatClient;
    // 词嵌入终端
    EmbeddingClientConfig embeddingClient;

};

/**
 * 模型配置
 */
struct ModelConfig {

    // 模型目录
    std::string modelPath;
    // 激活函数
    lifuren::config::Activation activation = lifuren::config::Activation::RELU;
    // 学习速率
    double learningRate = 0.0;
    // 正则函数
    lifuren::config::Regularization regularization = lifuren::config::Regularization::NONE;
    // 正则速率
    double regularizationRate = 0.0;

};

/**
 * 标记配置
 */
struct MarkConfig {

    // 目录
    std::string path;

    // 路径相同即为相等
    bool operator==(const std::string& path) const;

};

/**
 * 图片标记页面配置
 */
struct ImageMarkConfig : MarkConfig {
};

/**
 * 诗词标记页面配置
 */
struct PoetryMarkConfig : MarkConfig {
};

/**
 * 文档标记页面配置
 */
struct DocumentMarkConfig : MarkConfig {

    // RAG
    std::string rag;
    // 分段模型
    std::string chunk;
    // 词嵌入
    std::string embedding;

};

/**
 * ElasticSearch配置
 */
struct ElasticSearchConfig : RestConfig {

    // 词嵌入类型
    std::string embedding;

};

/**
 * 通用设置
 */
class Config {

public:
    lifuren::config::ChatConfig   chat  {};
    lifuren::config::ImageConfig  image {};
    lifuren::config::PoetryConfig poetry{};
    lifuren::config::OpenAiConfig openai{};
    lifuren::config::OllamaConfig ollama{};
    std::list<lifuren::config::ImageMarkConfig>    imageMark   {};
    std::list<lifuren::config::PoetryMarkConfig>   poetryMark  {};
    std::list<lifuren::config::DocumentMarkConfig> documentMark{};
    lifuren::config::ElasticSearchConfig elasticsearch{};

public:
    Config();
    virtual ~Config();

public:
    std::string toYaml();
    /**
     * @param T    配置泛型
     * @param name 配置名称
     * 
     * @return 配置指针
     */
    template<typename T>
    T* getConfig(const std::string& name) {
        if(CONFIG_CHAT == name) {
            return &this->chat;
        } else {
            return nullptr;
        }
    }

};

/**
 * 加载配置
 */
extern lifuren::config::Config loadFile();

/**
 * @param path 文件路径
 * 
 * @return 配置
 */
extern lifuren::config::Config loadFile(const std::string& path);

/**
 * 保存配置
 * 
 * @return 是否成功
 */
extern bool saveFile();

/**
 * @param path 文件路径
 * 
 * @return 是否成功
 */
extern bool saveFile(const std::string& path);

} // END OF lifuren::config

#endif // LFR_HEADER_CORE_CONFIG_CONFIG_HPP
