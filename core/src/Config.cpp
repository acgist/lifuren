#include "lifuren/Config.hpp"

#include <algorithm>
#include <filesystem>

#include "spdlog/spdlog.h"

#include "yaml-cpp/yaml.h"

#include "lifuren/File.hpp"
#include "lifuren/Yaml.hpp"

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

std::string lifuren::config::base           = "";
std::string lifuren::config::httpServerHost = "0.0.0.0";
int         lifuren::config::httpServerPort = 8080;

const std::string lifuren::config::CONFIG_CONFIG            = "config";
const std::string lifuren::config::CONFIG_HTTP_SERVER       = "http-server";
const std::string lifuren::config::CONFIG_AUDIO             = "audio";
const std::string lifuren::config::CONFIG_IMAGE             = "image";
const std::string lifuren::config::CONFIG_VIDEO             = "video";
const std::string lifuren::config::CONFIG_POETRY            = "poetry";
const std::string lifuren::config::CONFIG_MARK              = "mark";
const std::string lifuren::config::CONFIG_FAISS             = "faiss";
const std::string lifuren::config::CONFIG_ELASTICSEARCH     = "elasticsearch";
const std::string lifuren::config::CONFIG_OLLAMA            = "ollama";
const std::string lifuren::config::CONFIG_PEPPER            = "pepper";
const std::string lifuren::config::CONFIG_ACT_TANGXIANZU    = "act-tangxianzu";
const std::string lifuren::config::CONFIG_ACT_GUANHANQING   = "act-guanhanqing";
const std::string lifuren::config::CONFIG_PAINT_WUDAOZI     = "paint-wudaozi";
const std::string lifuren::config::CONFIG_PAINT_GUKAIZHI    = "paint-gukaizhi";
const std::string lifuren::config::CONFIG_COMPOSE_SHIKUANG  = "compose-shikuang";
const std::string lifuren::config::CONFIG_COMPOSE_LIGUINIAN = "compose-liguinian";
const std::string lifuren::config::CONFIG_POETIZE_LIDU      = "poetize-lidu";
const std::string lifuren::config::CONFIG_POETIZE_SUXIN     = "poetize-suxin";

lifuren::config::Config lifuren::config::CONFIG{};

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
    if(name == lifuren::config::CONFIG_CONFIG) {
        const auto& tmp = yaml["tmp"];
        if(tmp) {
            config.tmp = tmp.as<std::string>();
        }
    } else if(name == lifuren::config::CONFIG_HTTP_SERVER) {
        const auto& host = yaml["host"];
        const auto& port = yaml["port"];
        if(host) {
            lifuren::config::httpServerHost = host.as<std::string>();
        }
        if(port) {
            lifuren::config::httpServerPort = port.as<int>();
        }
    } else if(lifuren::config::CONFIG_AUDIO == name) {
        LFR_CONFIG_YAML_GETTER(config.audio, yaml, path,   path,   std::string);
        LFR_CONFIG_YAML_GETTER(config.audio, yaml, model,  model,  std::string);
        LFR_CONFIG_YAML_GETTER(config.audio, yaml, client, client, std::string);
        const YAML::Node& clients = yaml["clients"];
        if(clients) {
            std::for_each(clients.begin(), clients.end(), [&config](const auto& client) {
                config.audio.clients.emplace(client.template as<std::string>());
            });
        }
    } else if(lifuren::config::CONFIG_IMAGE == name) {
        LFR_CONFIG_YAML_GETTER(config.image, yaml, path,   path,   std::string);
        LFR_CONFIG_YAML_GETTER(config.image, yaml, model,  model,  std::string);
        LFR_CONFIG_YAML_GETTER(config.image, yaml, client, client, std::string);
        const YAML::Node& clients = yaml["clients"];
        if(clients) {
            std::for_each(clients.begin(), clients.end(), [&config](const auto& client) {
                config.image.clients.emplace(client.template as<std::string>());
            });
        }
    } else if(lifuren::config::CONFIG_VIDEO == name) {
        LFR_CONFIG_YAML_GETTER(config.video, yaml, path,   path,   std::string);
        LFR_CONFIG_YAML_GETTER(config.video, yaml, model,  model,  std::string);
        LFR_CONFIG_YAML_GETTER(config.video, yaml, length, length, int);
        LFR_CONFIG_YAML_GETTER(config.video, yaml, client, client, std::string);
        const YAML::Node& clients = yaml["clients"];
        if(clients) {
            std::for_each(clients.begin(), clients.end(), [&config](const auto& client) {
                config.video.clients.emplace(client.template as<std::string>());
            });
        }
    } else if(lifuren::config::CONFIG_POETRY == name) {
        LFR_CONFIG_YAML_GETTER(config.poetry, yaml, path,     path,     std::string);
        LFR_CONFIG_YAML_GETTER(config.poetry, yaml, model,    model,    std::string);
        LFR_CONFIG_YAML_GETTER(config.poetry, yaml, size,     size,     int);
        LFR_CONFIG_YAML_GETTER(config.poetry, yaml, length,   length,   int);
        LFR_CONFIG_YAML_GETTER(config.poetry, yaml, client,   client,   std::string);
        LFR_CONFIG_YAML_GETTER(config.poetry, yaml, rag-size, rag_size, int);
        LFR_CONFIG_YAML_GETTER(config.poetry, yaml, embedding-participle, embedding_participle, std::string);
        const YAML::Node& clients = yaml["clients"];
        if(clients) {
            std::for_each(clients.begin(), clients.end(), [&config](const auto& client) {
                config.poetry.clients.emplace(client.template as<std::string>());
            });
        }
    } else if(lifuren::config::CONFIG_ELASTICSEARCH == name) {
        LFR_CONFIG_YAML_GETTER(config.elasticsearch, yaml, api,       api,      std::string);
        LFR_CONFIG_YAML_GETTER(config.elasticsearch, yaml, username,  username, std::string);
        LFR_CONFIG_YAML_GETTER(config.elasticsearch, yaml, password,  password, std::string);
        LFR_CONFIG_YAML_GETTER(config.elasticsearch, yaml, auth-type, authType, std::string);
    } else if(lifuren::config::CONFIG_OLLAMA == name) {
        LFR_CONFIG_YAML_GETTER(config.ollama, yaml, api,       api,      std::string);
        LFR_CONFIG_YAML_GETTER(config.ollama, yaml, dims,      dims,     int);
        LFR_CONFIG_YAML_GETTER(config.ollama, yaml, username,  username, std::string);
        LFR_CONFIG_YAML_GETTER(config.ollama, yaml, password,  password, std::string);
        LFR_CONFIG_YAML_GETTER(config.ollama, yaml, auth-type, authType, std::string);
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
    } else if(lifuren::config::CONFIG_PEPPER == name) {
        LFR_CONFIG_YAML_GETTER(config.pepper, yaml, dims, dims, int);
        LFR_CONFIG_YAML_GETTER(config.pepper, yaml, path, path, std::string);
    } else {
        SPDLOG_DEBUG("配置没有适配加载：{}", name);
    }
}

static YAML::Node toYaml() {
    const auto& config = lifuren::config::CONFIG;
    YAML::Node yaml;
    {
        YAML::Node config;
        config["tmp"] = lifuren::config::CONFIG.tmp;
        yaml[lifuren::config::CONFIG_CONFIG] = config;
    }
    {
        YAML::Node http;
        http["host"] = lifuren::config::httpServerHost;
        http["port"] = lifuren::config::httpServerPort;
        yaml[lifuren::config::CONFIG_HTTP_SERVER] = http;
    }
    {
        YAML::Node audio;
        LFR_CONFIG_YAML_SETTER(audio, config.audio, path,   path);
        LFR_CONFIG_YAML_SETTER(audio, config.audio, model,  model);
        LFR_CONFIG_YAML_SETTER(audio, config.audio, client, client);
        YAML::Node clients;
        std::for_each(config.audio.clients.begin(), config.audio.clients.end(), [&clients](auto& v) {
            clients.push_back(v);
        });
        audio["clients"] = clients;
        yaml[lifuren::config::CONFIG_AUDIO] = audio;
    }
    {
        YAML::Node image;
        LFR_CONFIG_YAML_SETTER(image, config.image, path,   path);
        LFR_CONFIG_YAML_SETTER(image, config.image, model,  model);
        LFR_CONFIG_YAML_SETTER(image, config.image, client, client);
        YAML::Node clients;
        std::for_each(config.image.clients.begin(), config.image.clients.end(), [&clients](auto& v) {
            clients.push_back(v);
        });
        image["clients"] = clients;
        yaml[lifuren::config::CONFIG_IMAGE] = image;
    }
    {
        YAML::Node video;
        LFR_CONFIG_YAML_SETTER(video, config.video, path,   path);
        LFR_CONFIG_YAML_SETTER(video, config.video, model,  model);
        LFR_CONFIG_YAML_SETTER(video, config.video, length, length);
        LFR_CONFIG_YAML_SETTER(video, config.video, client, client);
        YAML::Node clients;
        std::for_each(config.video.clients.begin(), config.video.clients.end(), [&clients](auto& v) {
            clients.push_back(v);
        });
        video["clients"] = clients;
        yaml[lifuren::config::CONFIG_VIDEO] = video;
    }
    {
        YAML::Node poetry;
        LFR_CONFIG_YAML_SETTER(poetry, config.poetry, path,     path);
        LFR_CONFIG_YAML_SETTER(poetry, config.poetry, model,    model);
        LFR_CONFIG_YAML_SETTER(poetry, config.poetry, size,     size);
        LFR_CONFIG_YAML_SETTER(poetry, config.poetry, length,   length);
        LFR_CONFIG_YAML_SETTER(poetry, config.poetry, client,   client);
        LFR_CONFIG_YAML_SETTER(poetry, config.poetry, rag_size, rag-size);
        LFR_CONFIG_YAML_SETTER(poetry, config.poetry, embedding_participle, embedding-participle);
        YAML::Node clients;
        std::for_each(config.poetry.clients.begin(), config.poetry.clients.end(), [&clients](auto& v) {
            clients.push_back(v);
        });
        poetry["clients"] = clients;
        yaml[lifuren::config::CONFIG_POETRY] = poetry;
    }
    {
        YAML::Node elasticsearch;
        LFR_CONFIG_YAML_SETTER(elasticsearch, config.elasticsearch, api,       api);
        LFR_CONFIG_YAML_SETTER(elasticsearch, config.elasticsearch, username,  username);
        LFR_CONFIG_YAML_SETTER(elasticsearch, config.elasticsearch, password,  password);
        LFR_CONFIG_YAML_SETTER(elasticsearch, config.elasticsearch, authType,  auth-type);
        yaml[lifuren::config::CONFIG_ELASTICSEARCH] = elasticsearch;
    }
    {
        YAML::Node ollama;
        LFR_CONFIG_YAML_SETTER(ollama, config.ollama, api,      api);
        LFR_CONFIG_YAML_SETTER(ollama, config.ollama, dims,     dims);
        LFR_CONFIG_YAML_SETTER(ollama, config.ollama, username, username);
        LFR_CONFIG_YAML_SETTER(ollama, config.ollama, password, password);
        LFR_CONFIG_YAML_SETTER(ollama, config.ollama, authType, auth-type);
        YAML::Node embeddingClientNode;
        const lifuren::config::EmbeddingClientConfig& embeddingClient = config.ollama.embeddingClient;
        embeddingClientNode["path"]    = embeddingClient.path;
        embeddingClientNode["model"]   = embeddingClient.model;
        embeddingClientNode["options"] = embeddingClient.options;
        ollama["embedding"] = embeddingClientNode;
        yaml[lifuren::config::CONFIG_OLLAMA] = ollama;
    }
    {
        YAML::Node pepper;
        LFR_CONFIG_YAML_SETTER(pepper, config.pepper, dims, dims);
        LFR_CONFIG_YAML_SETTER(pepper, config.pepper, path, path);
        yaml[lifuren::config::CONFIG_PEPPER] = pepper;
    }
    return yaml;
}

lifuren::config::Config lifuren::config::loadFile() {
    try {
        return lifuren::config::loadFile(lifuren::config::baseFile(lifuren::config::CONFIG_PATH));
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
    YAML::Node yaml = lifuren::yaml::loadFile(path);
    if(!yaml || yaml.IsNull() || yaml.size() == 0LL) {
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
    return lifuren::config::saveFile(lifuren::config::baseFile(lifuren::config::CONFIG_PATH));
}

bool lifuren::config::saveFile(const std::string& path) {
    SPDLOG_INFO("保存配置文件：{}", path);
    return lifuren::yaml::saveFile(::toYaml(), path);
}

void lifuren::config::init(const int argc, const char* const argv[]) {
    if(argc > 0) {
        lifuren::config::base = std::filesystem::absolute(std::filesystem::u8path(argv[0]).parent_path()).string();
    }
    SPDLOG_DEBUG("执行目录：{}", lifuren::config::base);
    lifuren::config::loadConfig();
}

std::string lifuren::config::baseFile(const std::string& path) {
    return lifuren::file::join({lifuren::config::base, path}).string();
}

void lifuren::config::loadConfig() noexcept {
    // 配置
    lifuren::config::CONFIG = std::move(lifuren::config::loadFile());
    // 格律
    auto rhythm = lifuren::config::Rhythm::loadFile(lifuren::config::RHYTHM_PATH);
    lifuren::config::RHYTHM.clear();
    std::swap(lifuren::config::RHYTHM, rhythm);
    // lifuren::config::RHYTHM.insert(rhythm.begin(), rhythm.end());
}
