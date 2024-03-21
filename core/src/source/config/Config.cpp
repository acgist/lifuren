#include "../../header/config/Config.hpp"

std::string lifuren::config::httpServerHost = "0.0.0.0";
int         lifuren::config::httpServerPort = 8080;

std::map<std::string, lifuren::Config> lifuren::CONFIGS = lifuren::config::loadFile(lifuren::CONFIGS_PATH);

lifuren::Config::Config() {
}

lifuren::Config::~Config() {
}

lifuren::Config::Config(const YAML::Node& yaml) {
    if(yaml["modelPath"] && !yaml["modelPath"].IsNull()) {
        this->modelPath = yaml["modelPath"].as<std::string>();
    }
    if(yaml["datasetPath"] && !yaml["datasetPath"].IsNull()) {
        this->datasetPath = yaml["datasetPath"].as<std::string>();
    }
    if(yaml["activation"]) {
        this->activation = yaml["activation"].as<lifuren::Activation>();
    }
    if(yaml["learningRate"]) {
        this->learningRate = yaml["learningRate"].as<double>();
    }
    if(yaml["regularization"]) {
        this->regularization = yaml["regularization"].as<lifuren::Regularization>();
    }
    if(yaml["regularizationRate"]) {
        this->regularizationRate = yaml["regularizationRate"].as<double>();
    }
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
    for(
        auto iterator = yaml.begin();
        iterator != yaml.end();
        ++iterator
    ) {
        std::string key = iterator->first.as<std::string>();
        if(key == "httpServerHost") {
            lifuren::config::httpServerHost = iterator->second.as<std::string>();
        } else if(key == "httpServerPort") {
            lifuren::config::httpServerPort = iterator->second.as<int>();
        } else {
            Config config(iterator->second);
            map.insert(std::pair(key, config));
        }
    }
    return map;
}

bool lifuren::config::saveFile(const std::string& path) {
    SPDLOG_INFO("保存配置文件：{}", path);
    YAML::Node yaml;
    for(
        auto iterator = lifuren::CONFIGS.begin();
        iterator != lifuren::CONFIGS.end();
        ++iterator
    ) {
        yaml[iterator->first] = iterator->second.toYaml();
    }
    return lifuren::yamls::saveFile(yaml, path);
}
