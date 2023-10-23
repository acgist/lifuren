/**
 * 设置
 * 
 * @author acgist
 */
#pragma once

#include <map>
#include <string>

#include "GLog.hpp"
#include "nlohmann/json.hpp"

namespace lifuren {

/**
 * 激活函数
 */
enum Activation {
    RELU,
    TANH,
    LINEAR,
    SIGMOID,
};

/**
 * 正则函数
 */
enum Regularization {
    NONE,
    L1,
    L2,
};

/**
 * 设置
 */
class Setting {

public:
    // 训练文件路径
    std::string path;
    // 激活函数
    lifuren::Activation activation;
    // 学习速率
    double learningRate;
    // 正则函数
    lifuren::Regularization regularization;
    // 正则速率
    double regularizationRate;
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(lifuren::Setting, path, activation, learningRate, regularization, regularizationRate);
public:
    Setting();
    virtual ~Setting();
    /**
     * @param json JSON
     */
    Setting(const std::string& json);
    /**
     * @return JSON
     */
    virtual std::string toJSON();

};

/**
 * 设置
 */
class Settings {

public:
    // 配置
    std::map<std::string, lifuren::Setting> settings;

public:
    // 加载
    void load(const std::string& settings);

};

}