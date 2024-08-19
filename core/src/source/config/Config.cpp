#include "lifuren/config/Config.hpp"

#include <algorithm>

#include "spdlog/spdlog.h"

#include "lifuren/Yamls.hpp"
#include "lifuren/Logger.hpp"

// 配置读取
#ifndef LFR_CONFIG_YAML_GETTER
#define LFR_CONFIG_YAML_GETTER(config, yaml, key, name, type) \
const auto& name = yaml[#key];                                \
if(name && !name.IsNull() && name.IsScalar()) {               \
    config.name = name.template as<type>();                   \
}
#endif

// Map配置读取
#ifndef LFR_CONFIG_YAML_MAP_GETTER
#define LFR_CONFIG_YAML_MAP_GETTER(map, yaml, key, name, type)      \
const auto& name = yaml[#key];                                      \
if(name && !name.IsNull() && name.IsMap()) {                        \
    std::for_each(name.begin(), name.end(), [&map](const auto& v) { \
        const std::string& vk = v.first.template as<std::string>(); \
        const auto&        vv = v.second;                           \
        if(vv && !vv.IsNull() && vv.IsScalar()) {                   \
            map.emplace(vk, vv.template as<type>());                \
        }                                                           \
    });                                                             \
}
#endif

// List配置读取
#ifndef LFR_CONFIG_YAML_LIST_GETTER
#define LFR_CONFIG_YAML_LIST_GETTER(list, yaml, key, name, type)     \
const auto& name = yaml[#key];                                       \
if(name && !name.IsNull() && name.IsSequence()) {                    \
    std::for_each(name.begin(), name.end(), [&list](const auto& v) { \
        if(v && !v.IsNull() && v.IsScalar()) {                       \
            list.push_back(v.template as<std::string>());            \
        }                                                            \
    });                                                              \
}
#endif

// 配置写出
#ifndef LFR_CONFIG_YAML_SETTER
#define LFR_CONFIG_YAML_SETTER(target, source, name, key) \
target[#key] = source.name;
#endif

LFR_YAML_ENUM(config::Loss,           NONE, CROSS_ENTROPY_LOSS, NONE)
LFR_YAML_ENUM(config::Activation,     NONE, SOFTMAX,            NONE)
LFR_YAML_ENUM(config::Regularization, NONE, BATCH_NORM,         NONE)

LFR_LOG_FORMAT_ENUM(lifuren::config::Loss)
LFR_LOG_FORMAT_ENUM(lifuren::config::Activation)
LFR_LOG_FORMAT_ENUM(lifuren::config::Regularization)

const std::string lifuren::config::CONFIG_CHAT             = "chat";
const std::string lifuren::config::CONFIG_OPENAI           = "openai";
const std::string lifuren::config::CONFIG_OLLAMA           = "ollama";
const std::string lifuren::config::CONFIG_ELASTICSEARCH    = "elasticsearch";
const std::string lifuren::config::CONFIG_IMAGE_MARK       = "image-mark";
const std::string lifuren::config::CONFIG_POETRY_MARK      = "poetry-mark";
const std::string lifuren::config::CONFIG_DOCUMENT_MARK    = "document-mark";
const std::string lifuren::config::CONFIG_HTTP_SERVER_HOST = "http-server-host";
const std::string lifuren::config::CONFIG_HTTP_SERVER_PORT = "http-server-port";

std::string lifuren::config::httpServerHost = "0.0.0.0";
int         lifuren::config::httpServerPort = 8080;

/**
 * @param config 配置
 * @param name   配置名称
 * @param yaml   配置内容
 */
static void loadYaml(lifuren::config::Config& config, const std::string& name, const YAML::Node& yaml);
/**
 * @return YAML
 */
static YAML::Node toYaml();

lifuren::config::Config lifuren::config::CONFIG = lifuren::config::loadFile();

lifuren::config::Config::Config() {
}

lifuren::config::Config::~Config() {
}

std::string lifuren::config::Config::toYaml() {
    YAML::Node&& node = ::toYaml();
    std::stringstream stream;
    stream << node;
    return stream.str();
}

void loadYaml(lifuren::config::Config& config, const std::string& name, const YAML::Node& yaml) {
    if(name == lifuren::config::CONFIG_HTTP_SERVER_HOST) {
        lifuren::config::httpServerHost = yaml.as<std::string>();
    } else if(name == lifuren::config::CONFIG_HTTP_SERVER_PORT) {
        lifuren::config::httpServerPort = yaml.as<int>();
    } else if(lifuren::config::CONFIG_CHAT == name) {
        LFR_CONFIG_YAML_GETTER(config.chat, yaml, client,   client,  std::string);
        LFR_CONFIG_YAML_GETTER(config.chat, yaml, rag-size, ragSize, int);
        const YAML::Node& clients = yaml["clients"];
        if(clients) {
            std::for_each(clients.begin(), clients.end(), [&config](const auto& client) {
                config.chat.clients.emplace(client.template as<std::string>());
            });
        }
    } else if(lifuren::config::CONFIG_OPENAI == name) {
    } else if(lifuren::config::CONFIG_OLLAMA == name) {
        LFR_CONFIG_YAML_GETTER(config.ollama, yaml, api,       api,      std::string);
        LFR_CONFIG_YAML_GETTER(config.ollama, yaml, username,  username, std::string);
        LFR_CONFIG_YAML_GETTER(config.ollama, yaml, password,  password, std::string);
        LFR_CONFIG_YAML_GETTER(config.ollama, yaml, auth-type, authType, std::string);
        const YAML::Node& chatClientNode = yaml["chat"];
        if(chatClientNode) {
            lifuren::config::ChatClientConfig chatClient{};
            LFR_CONFIG_YAML_GETTER(chatClient, chatClientNode, path,        path,        std::string);
            LFR_CONFIG_YAML_GETTER(chatClient, chatClientNode, model,       model,       std::string);
            LFR_CONFIG_YAML_GETTER(chatClient, chatClientNode, top-p,       topP,        double);
            LFR_CONFIG_YAML_GETTER(chatClient, chatClientNode, top-k,       topK,        size_t);
            LFR_CONFIG_YAML_GETTER(chatClient, chatClientNode, temperature, temperature, double);
            std::map<std::string, std::string> map;
            LFR_CONFIG_YAML_MAP_GETTER(map, chatClientNode, options, options, std::string);
            chatClient.options.insert(map.begin(), map.end());
            config.ollama.chatClient = chatClient;
        }
        const YAML::Node& embeddingClientNode = yaml["embedding"];
        if(embeddingClientNode) {
            lifuren::config::EmbeddingClientConfig embeddingClient{};
            LFR_CONFIG_YAML_GETTER(embeddingClient, embeddingClientNode, path,    path,    std::string);
            LFR_CONFIG_YAML_GETTER(embeddingClient, embeddingClientNode, model,   model,   std::string);
            std::map<std::string, std::string> map;
            LFR_CONFIG_YAML_MAP_GETTER(map, embeddingClientNode, options, options, std::string);
            embeddingClient.options.insert(map.begin(), map.end());
            config.ollama.embeddingClient = embeddingClient;
        }
    } else if(lifuren::config::CONFIG_IMAGE_MARK == name) {
        std::for_each(yaml.begin(), yaml.end(), [&config](const auto& value) {
            lifuren::config::ImageMarkConfig imageMarkConfig{};
            LFR_CONFIG_YAML_GETTER(imageMarkConfig, value, path, path, std::string);
            config.imageMark.push_back(imageMarkConfig);
        });
    } else if(lifuren::config::CONFIG_POETRY_MARK == name) {
        std::for_each(yaml.begin(), yaml.end(), [&config](const auto& value) {
            lifuren::config::PoetryMarkConfig poetryMarkConfig{};
            LFR_CONFIG_YAML_GETTER(poetryMarkConfig, value, path, path, std::string);
            config.poetryMark.push_back(poetryMarkConfig);
        });
    } else if(lifuren::config::CONFIG_DOCUMENT_MARK == name) {
        std::for_each(yaml.begin(), yaml.end(), [&config](const auto& value) {
            lifuren::config::DocumentMarkConfig documentMarkConfig{};
            LFR_CONFIG_YAML_GETTER(documentMarkConfig, value, rag,       rag,       std::string);
            LFR_CONFIG_YAML_GETTER(documentMarkConfig, value, path,      path,      std::string);
            LFR_CONFIG_YAML_GETTER(documentMarkConfig, value, chunk,     chunk,     std::string);
            LFR_CONFIG_YAML_GETTER(documentMarkConfig, value, embedding, embedding, std::string);
            config.documentMark.push_back(documentMarkConfig);
        });
    } else if(lifuren::config::CONFIG_ELASTICSEARCH == name) {
        LFR_CONFIG_YAML_GETTER(config.elasticsearch, yaml, api,       api,      std::string);
        LFR_CONFIG_YAML_GETTER(config.elasticsearch, yaml, username,  username, std::string);
        LFR_CONFIG_YAML_GETTER(config.elasticsearch, yaml, password,  password, std::string);
        LFR_CONFIG_YAML_GETTER(config.elasticsearch, yaml, auth-type, authType, std::string);
        LFR_CONFIG_YAML_GETTER(config.elasticsearch, yaml, embedding, embedding, std::string);
    } else {
        SPDLOG_DEBUG("配置没有适配加载：{}", name);
    }
}

YAML::Node toYaml() {
    const auto& config = lifuren::config::CONFIG;
    YAML::Node yaml;
    {
        yaml[lifuren::config::CONFIG_HTTP_SERVER_HOST] = lifuren::config::httpServerHost;
        yaml[lifuren::config::CONFIG_HTTP_SERVER_PORT] = lifuren::config::httpServerPort;
    }
    {
        YAML::Node chat;
        LFR_CONFIG_YAML_SETTER(chat, config.chat, client,  client);
        LFR_CONFIG_YAML_SETTER(chat, config.chat, ragSize, rag-size);
        YAML::Node clients;
        std::for_each(config.chat.clients.begin(), config.chat.clients.end(), [&clients](auto& v) {
            clients.push_back(v);
        });
        chat["clients"] = clients;
        yaml[lifuren::config::CONFIG_CHAT] = chat;
    }
    {
        YAML::Node ollama;
        LFR_CONFIG_YAML_SETTER(ollama, config.ollama, api,      api);
        LFR_CONFIG_YAML_SETTER(ollama, config.ollama, username, username);
        LFR_CONFIG_YAML_SETTER(ollama, config.ollama, password, password);
        LFR_CONFIG_YAML_SETTER(ollama, config.ollama, authType, auth-type);
        YAML::Node chatClientNode;
        const lifuren::config::ChatClientConfig& chatClient = config.ollama.chatClient;
        chatClientNode["path"]  = chatClient.path;
        chatClientNode["model"] = chatClient.model;
        chatClientNode["top-p"] = chatClient.topP;
        chatClientNode["top-k"] = chatClient.topK;
        chatClientNode["temperature"] = chatClient.temperature;
        chatClientNode["options"] = chatClient.options;
        ollama["chat"] = chatClientNode;
        YAML::Node embeddingClientNode;
        const lifuren::config::EmbeddingClientConfig& embeddingClient = config.ollama.embeddingClient;
        embeddingClientNode["path"]    = embeddingClient.path;
        embeddingClientNode["model"]   = embeddingClient.model;
        embeddingClientNode["options"] = embeddingClient.options;
        ollama["embedding"] = embeddingClientNode;
        yaml[lifuren::config::CONFIG_OLLAMA] = ollama;
    }
    {
        YAML::Node imageMark;
        for(const auto& value : config.imageMark) {
            YAML::Node mark;
            mark["path"] = value.path;
            imageMark.push_back(mark);
        }
        yaml[lifuren::config::CONFIG_IMAGE_MARK] = imageMark;
    }
    {
        YAML::Node poetryMark;
        for(const auto& value : config.poetryMark) {
            YAML::Node mark;
            mark["path"] = value.path;
            poetryMark.push_back(mark);
        }
        yaml[lifuren::config::CONFIG_POETRY_MARK] = poetryMark;
    }
    {
        YAML::Node documentMark;
        for(const auto& value : config.documentMark) {
            YAML::Node mark;
            mark["rag"]       = value.rag;
            mark["path"]      = value.path;
            mark["chunk"]     = value.chunk;
            mark["embedding"] = value.embedding;
            documentMark.push_back(mark);
        }
        yaml[lifuren::config::CONFIG_DOCUMENT_MARK] = documentMark;
    }
    {
        YAML::Node elasticsearch;
        LFR_CONFIG_YAML_SETTER(elasticsearch, config.elasticsearch, api,       api);
        LFR_CONFIG_YAML_SETTER(elasticsearch, config.elasticsearch, username,  username);
        LFR_CONFIG_YAML_SETTER(elasticsearch, config.elasticsearch, password,  password);
        LFR_CONFIG_YAML_SETTER(elasticsearch, config.elasticsearch, authType,  auth-type);
        LFR_CONFIG_YAML_SETTER(elasticsearch, config.elasticsearch, embedding, embedding);
        yaml[lifuren::config::CONFIG_ELASTICSEARCH] = elasticsearch;
    }
    return yaml;
}

lifuren::config::Config lifuren::config::loadFile() {
    try {
        return lifuren::config::loadFile(lifuren::config::CONFIG_PATH);
    } catch(const std::exception& e) {
        SPDLOG_ERROR("加载配置异常：{}", e.what());
    } catch(...) {
        SPDLOG_ERROR("加载配置异常：未知原因");
    }
    return {};
}

lifuren::config::Config lifuren::config::loadFile(const std::string& path) {
    SPDLOG_DEBUG("加载配置文件：{}", path);
    lifuren::config::Config config{};
    YAML::Node yaml = lifuren::yamls::loadFile(path);
    if(!yaml || yaml.IsNull() || yaml.size() == 0L) {
        return config;
    }
    for(auto iterator = yaml.begin(); iterator != yaml.end(); ++iterator) {
        const std::string& key   = iterator->first.as<std::string>();
        const auto&        value = iterator->second;
        try {
            loadYaml(config, key, value);
        } catch(...) {
            SPDLOG_ERROR("加载配置异常：{}", key);
        }
    }
    return config;
}

bool lifuren::config::saveFile() {
    return lifuren::config::saveFile(lifuren::config::CONFIG_PATH);
}

bool lifuren::config::saveFile(const std::string& path) {
    SPDLOG_INFO("保存配置文件：{}", path);
    return lifuren::yamls::saveFile(toYaml(), path);
}

bool lifuren::config::MarkConfig::operator==(const std::string& path) const {
    return this->path == path;
}
