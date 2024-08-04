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
extern std::set<std::string> nlpClients;
// 聊天配置
extern const std::string CONFIG_CHAT;
// 训练数据集
extern const std::string CONFIG_DATASET;
// 自然语言终端列表
extern const std::string CONFIG_NLP_CLIENTS;
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
struct Chat {

    // 模型名称
    std::string model;
    // Embedding名称
    std::string embedding;

};

/**
 * 图片生成页面配置
 */
struct Image {

    // 模型名称
    std::string model;
    // Embedding名称
    std::string embedding;

};

/**
 * 视频生成页面配置
 */
struct Video {

    // 模型名称
    std::string model;
    // Embedding名称
    std::string embedding;

};

/**
 * REST Client配置
 */
struct RestClient {

    // 接口地址
    std::string api;
    // 账号
    std::string username;
    // 密码
    std::string password;

};

/**
 * OpenAi配置
 */
struct OpenAi : public RestClient {

};

/**
 * Ollama配置
 */
struct Ollama : public RestClient {

};

/**
 * 模型配置
 */
struct Model {

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
struct Dataset {

    // 数据集目录
    std::string datasetPath;

};

/**
 * 文档标记页面配置
 */
struct DocsMark : public Dataset {
};

/**
 * 图片标记页面配置
 */
struct ImageMark : public Dataset {
};

/**
 * 诗词标记页面配置
 */
struct PoetryMark : public Dataset {
};

/**
 * 通用设置
 */
class Config {

public:
    lifuren::config::Chat       chat      {};
    lifuren::config::Image      image     {};
    lifuren::config::Video      video     {};
    lifuren::config::OpenAi     openai    {};
    lifuren::config::Ollama     ollama    {};
    lifuren::config::DocsMark   docsMark  {};
    lifuren::config::ImageMark  imageMark {};
    lifuren::config::PoetryMark poetryMark{};

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
 * 模型配置
 */
class ModelConfig {

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
