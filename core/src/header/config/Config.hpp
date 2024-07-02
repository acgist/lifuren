/**
 * 系统设置
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_CONFIG_CONFIG_HPP
#define LFR_HEADER_CORE_CONFIG_CONFIG_HPP

#include <map>
#include <string>

#include "yaml-cpp/yaml.h"

namespace lifuren {

// 全局配置
class Config;

// 配置路径
const char* const CONFIGS_PATH = "../config/config.yml";

/**
 * 模型配置
 */
extern std::map<std::string, lifuren::Config> CONFIGS;

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
 * 模型设置
 */
class Config {

public:
    // 训练模型目录
    std::string modelPath;
    // 训练文件目录
    std::string datasetPath;
    // 激活函数
    lifuren::Activation activation = lifuren::Activation::RELU;
    // 学习速率
    double learningRate = 0.0;
    // 正则函数
    lifuren::Regularization regularization = lifuren::Regularization::NONE;
    // 正则速率
    double regularizationRate = 0.0;

public:
    Config();
    virtual ~Config();
    /**
     * @param yaml YAML
     */
    explicit Config(const YAML::Node& yaml);

public:
    /**
     * @return YAML
     */
    virtual YAML::Node toYaml();

};

/**
 * 模型配置
 */
class ModelConfig {

};

namespace model {

// 图片转为诗词
extern const std::string MODEL_I2P;
// 标签转为诗词
extern const std::string MODEL_L2P;
// 诗词转为图片
extern const std::string MODEL_P2I;
// 标签转为图片
extern const std::string MODEL_L2I;
// 图片转为图片
extern const std::string MODEL_I2I;
// 视频转为视频
extern const std::string MODEL_V2V;

} // END OF model

namespace config {

// 监听地址
extern std::string httpServerHost;
// 监听端口
extern int         httpServerPort;

/**
 * @param path 文件路径
 * 
 * @return 映射
 */
extern std::map<std::string, lifuren::Config> loadFile(const std::string& path);
/**
 * @param path 文件路径
 */
extern bool saveFile(const std::string& path);

} // END OF config
} // END OF lifuren

#endif // LFR_HEADER_CORE_CONFIG_CONFIG_HPP
