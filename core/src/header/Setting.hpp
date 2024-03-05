/**
 * 设置
 * 
 * @author acgist
 */
#pragma once

#include <map>
#include <string>

#include "Files.hpp"
#include "Logger.hpp"

#include "nlohmann/json.hpp"

namespace lifuren {

// 配置路径
const char* const SETTINGS_PATH = "../config/settings.json";

class Setting;

/**
 * 全局配置
 */
extern std::map<std::string, lifuren::Setting> SETTINGS;

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
class Setting {

public:
    // 训练模型目录
    std::string modelPath;
    // 训练文件目录
    std::string datasetPath;
    // 学习速率
    double learningRate = 0.0;
    // 正则速率
    double regularizationRate = 0.0;
    // 激活函数
    lifuren::Activation activation = lifuren::Activation::RELU;
    // 正则函数
    lifuren::Regularization regularization = lifuren::Regularization::NONE;
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Setting, modelPath, datasetPath, learningRate, regularizationRate, activation, regularization);

public:
    Setting();
    virtual ~Setting();
    /**
     * @param json JSON
     */
    explicit Setting(const std::string& json);

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON();

};

/**
 * 模型设置
 */
class ModelSetting {
};

}
