/**
 * 设置
 * 
 * @author acgist
 */
#pragma once

#include <map>
#include <string>

#include "../Logger.hpp"
#include "../utils/Yamls.hpp"

namespace lifuren {

class Config;

// 配置路径
const char* const CONFIGS_PATH = "../config/config.yml";

/**
 * 全局配置
 */
extern std::map<std::string, lifuren::Config> CONFIGS;

/**
 * 激活函数
 */
enum Activation {

    RELU,
    TANH,
    SIGMOID,

};

/**
 * 正则函数
 */
enum Regularization {

    NONE,
    L1,
    L2,
    MSE,

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

LFR_YAML_ENUM(Activation, RELU, SIGMOID, RELU)
LFR_YAML_ENUM(Regularization, NONE, MSE, NONE)
