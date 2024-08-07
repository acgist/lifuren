#include "lifuren/config/Config.hpp"

#include <algorithm>

#include "spdlog/spdlog.h"

#include "lifuren/Yamls.hpp"
#include "lifuren/Logger.hpp"

// 配置读取
#ifndef LFR_CONFIG_YAML_GETTER
#define LFR_CONFIG_YAML_GETTER(config, yaml, key, name, type) \
auto& name = yaml[#key];                                      \
if(name && !name.IsNull()) {                                  \
    this->config.name = name.as<type>();                      \
}
#endif

// 配置写出
#ifndef LFR_CONFIG_YAML_SETTER
#define LFR_CONFIG_YAML_SETTER(config, name, key) \
config[#key] = this->config.name;
#endif

LFR_YAML_ENUM(config::Loss,           NONE, CROSS_ENTROPY_LOSS, NONE)
LFR_YAML_ENUM(config::Activation,     NONE, SOFTMAX,            NONE)
LFR_YAML_ENUM(config::Regularization, NONE, BATCH_NORM,         NONE)

LFR_LOG_FORMAT_ENUM(lifuren::config::Loss)
LFR_LOG_FORMAT_ENUM(lifuren::config::Activation)
LFR_LOG_FORMAT_ENUM(lifuren::config::Regularization)

const std::string lifuren::config::CONFIG_CHAT             = "chat";
const std::string lifuren::config::CONFIG_DATASET          = "dataset";
const std::string lifuren::config::CONFIG_CHAT_CLIENTS     = "chat-clients";
const std::string lifuren::config::CONFIG_OPENAI           = "openai";
const std::string lifuren::config::CONFIG_OLLAMA           = "ollama";
const std::string lifuren::config::CONFIG_HTTP_SERVER_HOST = "http-server-host";
const std::string lifuren::config::CONFIG_HTTP_SERVER_PORT = "http-server-port";

std::string lifuren::config::httpServerHost = "0.0.0.0";
int         lifuren::config::httpServerPort = 8080;
std::set<std::string> lifuren::config::chatClients{};

lifuren::config::Config lifuren::config::CONFIG = lifuren::config::loadFile(lifuren::config::CONFIG_PATH);

lifuren::config::Config::Config() {
}

lifuren::config::Config::~Config() {
}

void lifuren::config::Config::loadYaml(const std::string& name, const YAML::Node& yaml) {
    if(CONFIG_CHAT == name) {
        LFR_CONFIG_YAML_GETTER(chat, yaml, client, client, std::string);
    } else if(CONFIG_OPENAI == name) {
    } else if(CONFIG_OLLAMA == name) {
        LFR_CONFIG_YAML_GETTER(ollama, yaml, api,       api,      std::string);
        LFR_CONFIG_YAML_GETTER(ollama, yaml, username,  username, std::string);
        LFR_CONFIG_YAML_GETTER(ollama, yaml, password,  password, std::string);
        LFR_CONFIG_YAML_GETTER(ollama, yaml, auth-type, authType, std::string);
    } else {
        SPDLOG_DEBUG("配置没有适配加载：{}", name);
    }
}

YAML::Node lifuren::config::Config::toYaml() {
    YAML::Node yaml;
    YAML::Node chat;
    LFR_CONFIG_YAML_SETTER(chat, client, client);
    yaml[CONFIG_CHAT] = chat;
    YAML::Node ollama;
    LFR_CONFIG_YAML_SETTER(ollama, api,      api);
    LFR_CONFIG_YAML_SETTER(ollama, username, username);
    LFR_CONFIG_YAML_SETTER(ollama, password, password);
    LFR_CONFIG_YAML_SETTER(ollama, authType, auth-type);
    yaml[CONFIG_OLLAMA] = ollama;
    YAML::Node chatClients;
    std::for_each(lifuren::config::chatClients.begin(), lifuren::config::chatClients.end(), [&chatClients](auto& v) {
        chatClients.push_back(v);
    });
    yaml[CONFIG_CHAT_CLIENTS] = chatClients;
    return yaml;
}

inline lifuren::config::Config lifuren::config::loadFile() {
    return lifuren::config::loadFile(lifuren::config::CONFIG_PATH);
}

lifuren::config::Config lifuren::config::loadFile(const std::string& path) {
    SPDLOG_DEBUG("加载配置文件：{}", path);
    lifuren::config::Config config{};
    YAML::Node yaml = lifuren::yamls::loadFile(path);
    if(!yaml || yaml.IsNull() || yaml.size() == 0L) {
        return config;
    }
    for(auto iterator = yaml.begin(); iterator != yaml.end(); ++iterator) {
        std::string key   = iterator->first.as<std::string>();
        auto&       value = iterator->second;
        if(key == CONFIG_HTTP_SERVER_HOST) {
            lifuren::config::httpServerHost = value.as<std::string>();
        } else if(key == CONFIG_HTTP_SERVER_PORT) {
            lifuren::config::httpServerPort = value.as<int>();
        } else if(key == CONFIG_CHAT_CLIENTS) {
            std::for_each(value.begin(), value.end(), [](auto client) {
                lifuren::config::chatClients.emplace(client.as<std::string>());
            });
        } else {
            config.loadYaml(key, value);
        }
    }
    return config;
}

inline bool lifuren::config::saveFile() {
    return lifuren::config::saveFile(lifuren::config::CONFIG_PATH);
}

bool lifuren::config::saveFile(const std::string& path) {
    SPDLOG_INFO("保存配置文件：{}", path);
    return lifuren::yamls::saveFile(CONFIG.toYaml(), path);
}
