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

extern const std::string CONFIG_HTTP_SERVER;
extern const std::string CONFIG_IMAGE;
extern const std::string CONFIG_POETRY;
extern const std::string CONFIG_IMAGE_MARK;
extern const std::string CONFIG_POETRY_MARK;
extern const std::string CONFIG_RAG;
extern const std::string CONFIG_EMBEDDING;
extern const std::string CONFIG_OLLAMA;
extern const std::string CONFIG_ELASTICSEARCH;
extern const std::string CONFIG_POETIZE_RNN;
extern const std::string CONFIG_PAINT_CYCLE_GAN;
extern const std::string CONFIG_PAINT_STYLE_GAN;
extern const std::string CONFIG_STABLE_DIFFUSION_CPP;
extern const std::string CONFIG_CHINESE_WORD_VECTORS;

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
 * 命令配置
 */
struct CommandConfig {

    // 命令路径
    std::string command;
    // 默认参数
    std::map<std::string, std::string> options{};

};

/**
 * 词嵌入终端配置
 */
struct EmbeddingClientConfig {

    // 请求地址
    std::string path;
    // 词嵌入模型
    std::string model;
    // 其他配置
    std::map<std::string, std::string> options{};

};

/**
 * 图片生成页面配置
 */
struct ImageConfig {

    // 终端名称
    std::string client;
    // 输出目录
    std::string output;
    // 终端列表
    std::set<std::string> clients;

};

/**
 * 诗词生成页面配置
 */
struct PoetryConfig {

    // 终端名称
    std::string client;
    // 终端列表
    std::set<std::string> clients;

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
 * RAG配置
 */
struct RAGConfig {

    // 最后索引
    size_t id;
    // RAG类型
    std::string type;
    // 返回数量
    size_t size;

};

/**
 * 词嵌入配置
 */
struct EmbeddingConfig {

    // 词嵌入类型
    std::string type;

};

/**
 * Ollama配置
 */
struct OllamaConfig : RestConfig {

    // 词嵌入终端
    EmbeddingClientConfig embeddingClient;

};

/**
 * ElasticSearch配置
 */
struct ElasticSearchConfig : RestConfig {

};

/**
 * chinese-word-vectors配置
 */
struct ChineseWordVectorsConfig {

    std::string path;

};

/**
 * poetize-rnn配置
 */
struct PoetizeRNNConfig {
    
    // 模型路径
    std::string model;

};

/**
 * paint-cycle-gan配置
 */
struct PaintCycleGANConfig {
    
    // 模型路径
    std::string model;

};

/**
 * paint-style-gan配置
 */
struct PaintStyleGANConfig {
    
    // 模型路径
    std::string model;

};

/**
 * stable-diffusion-cpp配置
 */
struct StableDiffusionCPPConfig {

    // 模型路径
    std::string model;
    // 默认参数
    std::map<std::string, std::string> options{};

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
 * 通用设置
 */
class Config {

public:
    lifuren::config::ImageConfig  image {};
    lifuren::config::PoetryConfig poetry{};
    lifuren::config::OllamaConfig ollama{};
    std::list<lifuren::config::ImageMarkConfig>  imageMark {};
    std::list<lifuren::config::PoetryMarkConfig> poetryMark{};
    lifuren::config::RAGConfig       rag{};
    lifuren::config::EmbeddingConfig embedding{};
    lifuren::config::ElasticSearchConfig      elasticsearch{};
    lifuren::config::ChineseWordVectorsConfig chineseWordVectors{};
    lifuren::config::PoetizeRNNConfig         poetizeRNN{};
    lifuren::config::PaintCycleGANConfig      paintCycleGAN{};
    lifuren::config::PaintStyleGANConfig      paintSytleGAN{};
    lifuren::config::StableDiffusionCPPConfig stableDiffusionCPP{};

public:
    Config();
    virtual ~Config();

public:
    std::string toYaml();

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
