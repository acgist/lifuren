#include "lifuren/config/Config.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"
#include "lifuren/utils/Yamls.hpp"

// 配置读取
#ifndef LFR_CONFIG_YAML_GETTER
#define LFR_CONFIG_YAML_GETTER(yaml, name, type) \
auto& name = yaml[#name];                        \
if(name && !name.IsNull()) {                     \
    this->name = name.as<type>();                \
}
#endif

LFR_YAML_ENUM(Loss,           NONE, CROSS_ENTROPY_LOSS, NONE)
LFR_YAML_ENUM(Activation,     NONE, SOFTMAX,            NONE)
LFR_YAML_ENUM(Regularization, NONE, BATCH_NORM,         NONE)

LFR_LOG_FORMAT_ENUM(lifuren::Loss)
LFR_LOG_FORMAT_ENUM(lifuren::Activation)
LFR_LOG_FORMAT_ENUM(lifuren::Regularization)

const std::string lifuren::model::MODEL_I2P = "i2p";
const std::string lifuren::model::MODEL_L2P = "l2p";
const std::string lifuren::model::MODEL_P2I = "p2i";
const std::string lifuren::model::MODEL_L2I = "l2i";
const std::string lifuren::model::MODEL_I2I = "i2i";
const std::string lifuren::model::MODEL_V2V = "v2v";

const std::string lifuren::config::CONFIG_DATASET = "dataset";

std::string lifuren::config::httpServerHost = "0.0.0.0";
int         lifuren::config::httpServerPort = 8080;

std::map<std::string, lifuren::Config> lifuren::CONFIGS = lifuren::config::loadFile(lifuren::CONFIGS_PATH);

lifuren::Config::Config() {
}

lifuren::Config::~Config() {
}

lifuren::Config::Config(const YAML::Node& yaml) {
    LFR_CONFIG_YAML_GETTER(yaml, modelPath,          std::string);
    LFR_CONFIG_YAML_GETTER(yaml, datasetPath,        std::string);
    LFR_CONFIG_YAML_GETTER(yaml, activation,         lifuren::Activation);
    LFR_CONFIG_YAML_GETTER(yaml, learningRate,       double);
    LFR_CONFIG_YAML_GETTER(yaml, regularization,     lifuren::Regularization);
    LFR_CONFIG_YAML_GETTER(yaml, regularizationRate, double);
}

YAML::Node lifuren::Config::toYaml() {
    YAML::Node yaml;
    yaml["modelPath"]          = this->modelPath;
    yaml["datasetPath"]        = this->datasetPath;
    yaml["activation"]         = this->activation;
    yaml["learningRate"]       = this->learningRate;
    yaml["regularization"]     = this->regularization;
    yaml["regularizationRate"] = this->regularizationRate;
    return yaml;
}

std::map<std::string, lifuren::Config> lifuren::config::loadFile(const std::string& path) {
    SPDLOG_DEBUG("加载配置文件：{}", path);
    std::map<std::string, lifuren::Config> map;
    YAML::Node yaml = lifuren::yamls::loadFile(path);
    if(yaml.size() == 0L) {
        return map;
    }
    for(auto iterator = yaml.begin(); iterator != yaml.end(); ++iterator) {
        std::string key = iterator->first.as<std::string>();
        if(key == "httpServerHost") {
            lifuren::config::httpServerHost = iterator->second.as<std::string>();
        } else if(key == "httpServerPort") {
            lifuren::config::httpServerPort = iterator->second.as<int>();
        } else {
            Config config(iterator->second);
            map.emplace(key, config);
        }
    }
    return map;
}

bool lifuren::config::saveFile(const std::string& path) {
    SPDLOG_INFO("保存配置文件：{}", path);
    YAML::Node yaml;
    for(auto iterator = lifuren::CONFIGS.begin(); iterator != lifuren::CONFIGS.end(); ++iterator) {
        yaml[iterator->first] = iterator->second.toYaml();
    }
    return lifuren::yamls::saveFile(yaml, path);
}
