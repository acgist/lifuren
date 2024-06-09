/**
 * 设置
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_CONFIG_CONFIG_HPP
#define LFR_HEADER_CORE_CONFIG_CONFIG_HPP

#include <map>
#include <string>

#include "Logger.hpp"
#include "../utils/Yamls.hpp"

#include "spdlog/fmt/ostr.h"
#include "spdlog/fmt/chrono.h"
#include "spdlog/fmt/ranges.h"

namespace lifuren {

class Config;

// 配置路径
const char* const CONFIGS_PATH = "../config/config.yml";

/**
 * 全局配置
 */
extern std::map<std::string, lifuren::Config> CONFIGS;

/**
 * 损失函数
 */
enum class Loss {

    NONE               = 0,
    MSE                = 1,
    NLL                = 2,
    CROSS_ENTROPY_LOSS = 3,

};

/**
 * 激活函数
 */
enum class Activation {

    NONE    = 0,
    RELU    = 1,
    TANH    = 2,
    SIGMOID = 3,
    SOFTMAX = 4,

};

/**
 * 正则函数
 */
enum class Regularization {

    NONE       = 0,
    L1         = 1,
    L2         = 2,
    DROPOUT    = 3,
    BATCH_NORM = 4,

};


/**
 * 设置
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

}
}

LFR_YAML_ENUM(Loss, NONE, CROSS_ENTROPY_LOSS, NONE)
LFR_YAML_ENUM(Activation, NONE, SOFTMAX, RELU)
LFR_YAML_ENUM(Regularization, NONE, BATCH_NORM, NONE)

LFR_LOG_FORMAT_ENUM(lifuren::Loss)
LFR_LOG_FORMAT_ENUM(lifuren::Activation)
LFR_LOG_FORMAT_ENUM(lifuren::Regularization)

#endif // LFR_HEADER_CORE_CONFIG_CONFIG_HPP
