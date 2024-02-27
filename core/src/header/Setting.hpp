/**
 * 设置
 * 
 * @author acgist
 */
#pragma once

#include <map>
#include <string>
#include <fstream>

#include "Logger.hpp"

#include "nlohmann/json.hpp"

namespace lifuren {

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
    // 训练模型路径
    std::string modelPath;
    // 训练文件路径
    std::string datasetPath;
    // 激活函数
    lifuren::Activation activation = lifuren::Activation::RELU;
    // 学习速率
    double learningRate = 0.0;
    // 正则函数
    lifuren::Regularization regularization = lifuren::Regularization::NONE;
    // 正则速率
    double regularizationRate = 0.0;
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Setting, modelPath, datasetPath, activation, learningRate, regularization, regularizationRate);

public:
    Setting();
    virtual ~Setting();
    /**
     * @param json JSON
     */
    Setting(const std::string& json);

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON();

};

/**
 * 设置集合
 */
class Settings {

public:
    // 设置
    std::map<std::string, Setting> settings;
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Settings, settings);

public:
    Settings();
    ~Settings();

public:
    /**
     * 加载设置
     * 
     * @param settings JSON
     */
    void load(const std::string& settings);
    /**
     * 加载文件设置
     * 
     * @param path 文件路径
     */
    void loadFile(const std::string& path);
    /**
     * 保存文件
     * 
     * @param path 文件路径
     */
    void saveFile(const std::string& path);
    /**
     * @return JSON
     */
    std::string toJSON();

};

/**
 * 预测设置
 */
class PredictSetting {
};

/**
 * 训练设置
 */
class TrainingSetting {
};

// 配置路径
static const char* SETTINGS_PATH = "../config/settings.json";

/**
 * 全局配置
 */
extern lifuren::Settings SETTINGS;

}
