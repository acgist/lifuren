/**
 * 系统设置
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_CONFIG_CONFIG_HPP
#define LFR_HEADER_CORE_CONFIG_CONFIG_HPP

#include <set>
#include <string>

#include "yaml-cpp/yaml.h"

namespace lifuren::config {

// 前置声明
class Config;

// 监听地址
extern std::string httpServerHost;
// 监听端口
extern int         httpServerPort;
// 自然语言终端列表
extern std::set<std::string> chatClients;
// 聊天配置
extern const std::string CONFIG_CHAT;
// 训练数据集
extern const std::string CONFIG_DATASET;
// 自然语言终端列表
extern const std::string CONFIG_CHAT_CLIENTS;
// OpenAI
extern const std::string CONFIG_OPENAI;
// Ollama
extern const std::string CONFIG_OLLAMA;
// 监听地址
extern const std::string CONFIG_HTTP_SERVER_HOST;
// 监听端口
extern const std::string CONFIG_HTTP_SERVER_PORT;
// 配置路径
const char* const CONFIG_PATH = "../config/config.yml";
// 全局配置
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

    // 终端名称
    std::string client;

};

/**
 * 图片生成页面配置
 */
struct ImageConfig {

};

/**
 * 视频生成页面配置
 */
struct VideoConfig {

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

};

/**
 * LLM配置
 */
struct LLMConfig {

    double topP;
    size_t topK;
    double temperature;
    std::string options{ "{}" };

};

/**
 * 聊天终端
 */
struct ChatClientConfig : public LLMConfig {

    std::string path;
    std::string model;

};

/**
 * 词嵌入终端
 */
struct EmbeddingClientConfig {

    std::string path;
    std::string model;
    std::string options{ "{}" };

};

/**
 * OpenAi配置
 */
struct OpenAiConfig : public RestConfig {

    // 聊天终端
    ChatClientConfig chatClient;
    // 词嵌入终端
    EmbeddingClientConfig embeddingClient;

};

/**
 * Ollama配置
 */
struct OllamaConfig : public RestConfig {

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
 * 数据集配置
 */
struct DatasetConfig {

    // 数据集目录
    std::string datasetPath;

};

/**
 * 文档标记页面配置
 */
struct DocsMarkConfig : public DatasetConfig {
};

/**
 * 图片标记页面配置
 */
struct ImageMarkConfig : public DatasetConfig {
};

/**
 * 诗词标记页面配置
 */
struct PoetryMarkConfig : public DatasetConfig {
};

/**
 * 通用设置
 */
class Config {

public:
    lifuren::config::ChatConfig       chat      {};
    lifuren::config::ImageConfig      image     {};
    lifuren::config::VideoConfig      video     {};
    lifuren::config::OpenAiConfig     openai    {};
    lifuren::config::OllamaConfig     ollama    {};
    lifuren::config::DocsMarkConfig   docsMark  {};
    lifuren::config::ImageMarkConfig  imageMark {};
    lifuren::config::PoetryMarkConfig poetryMark{};

public:
    Config();
    virtual ~Config();

public:
    /**
     * @param name 配置名称
     * @param yaml 配置内容
     */
    void loadYaml(const std::string& name, const YAML::Node& yaml);
    /**
     * @return YAML
     */
    YAML::Node toYaml();
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
extern inline lifuren::config::Config loadFile();

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
extern inline bool saveFile();

/**
 * @param path 文件路径
 * 
 * @return 是否成功
 */
extern bool saveFile(const std::string& path);

} // END OF lifuren::config

#endif // LFR_HEADER_CORE_CONFIG_CONFIG_HPP
