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

const std::string lifuren::config::CONFIG_CONFIG               = "config";
const std::string lifuren::config::CONFIG_HTTP_SERVER          = "http-server";
const std::string lifuren::config::CONFIG_IMAGE                = "image";
const std::string lifuren::config::CONFIG_POETRY               = "poetry";
const std::string lifuren::config::CONFIG_MARK                 = "mark";
const std::string lifuren::config::CONFIG_RAG                  = "rag";
const std::string lifuren::config::CONFIG_EMBEDDING            = "embedding";
const std::string lifuren::config::CONFIG_OLLAMA               = "ollama";
const std::string lifuren::config::CONFIG_ELASTICSEARCH        = "elasticsearch";
const std::string lifuren::config::CONFIG_CHINESE_WORD_VECTORS = "chinese-word-vectors";
const std::string lifuren::config::CONFIG_POETIZE_SHIFO_RNN    = "poetize-shifo-rnn";
const std::string lifuren::config::CONFIG_POETIZE_SHIMO_RNN    = "poetize-shimo-rnn";
const std::string lifuren::config::CONFIG_POETIZE_SHIGUI_RNN   = "poetize-shigui-rnn";
const std::string lifuren::config::CONFIG_POETIZE_SHIXIAN_RNN  = "poetize-shixian-rnn";
const std::string lifuren::config::CONFIG_POETIZE_SHISHENG_RNN = "poetize-shisheng-rnn";
const std::string lifuren::config::CONFIG_POETIZE_LIDU_RNN     = "poetize-lidu-rnn";
const std::string lifuren::config::CONFIG_POETIZE_SUXIN_RNN    = "poetize-suxin-rnn";
const std::string lifuren::config::CONFIG_POETIZE_WANYUE_RNN   = "poetize-wanyue-rnn";
const std::string lifuren::config::CONFIG_PAINT_CYCLE_GAN      = "paint-cycle-gan";
const std::string lifuren::config::CONFIG_PAINT_STYLE_GAN      = "paint-style-gan";
const std::string lifuren::config::CONFIG_STABLE_DIFFUSION_CPP = "stable-diffusion-cpp";

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
    } else if(lifuren::config::CONFIG_IMAGE == name) {
        LFR_CONFIG_YAML_GETTER(config.image, yaml, client, client, std::string);
        LFR_CONFIG_YAML_GETTER(config.image, yaml, output, output, std::string);
        const YAML::Node& clients = yaml["clients"];
        if(clients) {
            std::for_each(clients.begin(), clients.end(), [&config](const auto& client) {
                config.image.clients.emplace(client.template as<std::string>());
            });
        }
    } else if(lifuren::config::CONFIG_POETRY == name) {
        LFR_CONFIG_YAML_GETTER(config.poetry, yaml, client, client, std::string);
        const YAML::Node& clients = yaml["clients"];
        if(clients) {
            std::for_each(clients.begin(), clients.end(), [&config](const auto& client) {
                config.poetry.clients.emplace(client.template as<std::string>());
            });
        }
    } else if(lifuren::config::CONFIG_MARK == name) {
        std::for_each(yaml.begin(), yaml.end(), [&config](const auto& value) {
            lifuren::config::MarkConfig markConfig{};
            LFR_CONFIG_YAML_GETTER(markConfig, value, path, path, std::string);
            config.mark.push_back(markConfig);
        });
    } else if(lifuren::config::CONFIG_RAG == name) {
        LFR_CONFIG_YAML_GETTER(config.rag, yaml, type, type, std::string);
        LFR_CONFIG_YAML_GETTER(config.rag, yaml, size, size, size_t);
    } else if(lifuren::config::CONFIG_EMBEDDING == name) {
        LFR_CONFIG_YAML_GETTER(config.embedding, yaml, type,       type,       std::string);
        LFR_CONFIG_YAML_GETTER(config.embedding, yaml, participle, participle, std::string);
    } else if(lifuren::config::CONFIG_OLLAMA == name) {
        LFR_CONFIG_YAML_GETTER(config.ollama, yaml, api,       api,      std::string);
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
    } else if(lifuren::config::CONFIG_ELASTICSEARCH == name) {
        LFR_CONFIG_YAML_GETTER(config.elasticsearch, yaml, api,       api,      std::string);
        LFR_CONFIG_YAML_GETTER(config.elasticsearch, yaml, username,  username, std::string);
        LFR_CONFIG_YAML_GETTER(config.elasticsearch, yaml, password,  password, std::string);
        LFR_CONFIG_YAML_GETTER(config.elasticsearch, yaml, auth-type, authType, std::string);
    } else if(lifuren::config::CONFIG_CHINESE_WORD_VECTORS == name) {
        LFR_CONFIG_YAML_GETTER(config.chineseWordVectors, yaml, path, path, std::string);
    } else if(lifuren::config::CONFIG_POETIZE_SHIFO_RNN == name) {
        LFR_CONFIG_YAML_GETTER(config.poetizeShifoRNN, yaml, model, model, std::string);
    } else if(lifuren::config::CONFIG_POETIZE_SHIMO_RNN == name) {
        LFR_CONFIG_YAML_GETTER(config.poetizeShimoRNN, yaml, model, model, std::string);
    } else if(lifuren::config::CONFIG_POETIZE_SHIGUI_RNN == name) {
        LFR_CONFIG_YAML_GETTER(config.poetizeShiguiRNN, yaml, model, model, std::string);
    } else if(lifuren::config::CONFIG_POETIZE_SHIXIAN_RNN == name) {
        LFR_CONFIG_YAML_GETTER(config.poetizeShixianRNN, yaml, model, model, std::string);
    } else if(lifuren::config::CONFIG_POETIZE_SHISHENG_RNN == name) {
        LFR_CONFIG_YAML_GETTER(config.poetizeShishengRNN, yaml, model, model, std::string);
    } else if(lifuren::config::CONFIG_POETIZE_LIDU_RNN == name) {
        LFR_CONFIG_YAML_GETTER(config.poetizeLiduRNN, yaml, model, model, std::string);
    } else if(lifuren::config::CONFIG_POETIZE_SUXIN_RNN == name) {
        LFR_CONFIG_YAML_GETTER(config.poetizeSuxinRNN, yaml, model, model, std::string);
    } else if(lifuren::config::CONFIG_POETIZE_WANYUE_RNN == name) {
        LFR_CONFIG_YAML_GETTER(config.poetizeWanyueRNN, yaml, model, model, std::string);
    } else if(lifuren::config::CONFIG_PAINT_CYCLE_GAN == name) {
        LFR_CONFIG_YAML_GETTER(config.paintCycleGAN, yaml, model, model, std::string);
    } else if(lifuren::config::CONFIG_PAINT_STYLE_GAN == name) {
        LFR_CONFIG_YAML_GETTER(config.paintSytleGAN, yaml, model, model, std::string);
    } else if(lifuren::config::CONFIG_STABLE_DIFFUSION_CPP == name) {
        LFR_CONFIG_YAML_GETTER(config.stableDiffusionCPP, yaml, model, model, std::string);
        std::map<std::string, std::string> map;
        LFR_CONFIG_YAML_MAP_GETTER(map, yaml, options, options, std::string);
        config.stableDiffusionCPP.options.insert(map.begin(), map.end());
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
        YAML::Node image;
        LFR_CONFIG_YAML_SETTER(image, config.image, client, client);
        LFR_CONFIG_YAML_SETTER(image, config.image, output, output);
        YAML::Node clients;
        std::for_each(config.image.clients.begin(), config.image.clients.end(), [&clients](auto& v) {
            clients.push_back(v);
        });
        image["clients"] = clients;
        yaml[lifuren::config::CONFIG_IMAGE] = image;
    }
    {
        YAML::Node poetry;
        LFR_CONFIG_YAML_SETTER(poetry, config.poetry, client, client);
        YAML::Node clients;
        std::for_each(config.poetry.clients.begin(), config.poetry.clients.end(), [&clients](auto& v) {
            clients.push_back(v);
        });
        poetry["clients"] = clients;
        yaml[lifuren::config::CONFIG_POETRY] = poetry;
    }
    {
        YAML::Node mark;
        for(const auto& value : config.mark) {
            YAML::Node item;
            item["path"] = value.path;
            mark.push_back(item);
        }
        yaml[lifuren::config::CONFIG_MARK] = mark;
    }
    {
        YAML::Node rag;
        LFR_CONFIG_YAML_SETTER(rag, config.rag, type, type);
        LFR_CONFIG_YAML_SETTER(rag, config.rag, size, size);
        yaml[lifuren::config::CONFIG_RAG] = rag;
    }
    {
        YAML::Node embedding;
        LFR_CONFIG_YAML_SETTER(embedding, config.embedding, type,       type);
        LFR_CONFIG_YAML_SETTER(embedding, config.embedding, participle, participle);
        yaml[lifuren::config::CONFIG_EMBEDDING] = embedding;
    }
    {
        YAML::Node ollama;
        LFR_CONFIG_YAML_SETTER(ollama, config.ollama, api,      api);
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
        YAML::Node elasticsearch;
        LFR_CONFIG_YAML_SETTER(elasticsearch, config.elasticsearch, api,       api);
        LFR_CONFIG_YAML_SETTER(elasticsearch, config.elasticsearch, username,  username);
        LFR_CONFIG_YAML_SETTER(elasticsearch, config.elasticsearch, password,  password);
        LFR_CONFIG_YAML_SETTER(elasticsearch, config.elasticsearch, authType,  auth-type);
        yaml[lifuren::config::CONFIG_ELASTICSEARCH] = elasticsearch;
    }
    {
        YAML::Node chineseWordVectors;
        LFR_CONFIG_YAML_SETTER(chineseWordVectors, config.chineseWordVectors, path, path);
        yaml[lifuren::config::CONFIG_CHINESE_WORD_VECTORS] = chineseWordVectors;
    }
    {
        YAML::Node poetizeShifoRNN;
        LFR_CONFIG_YAML_SETTER(poetizeShifoRNN, config.poetizeShifoRNN, model, model);
        yaml[lifuren::config::CONFIG_POETIZE_SHIFO_RNN] = poetizeShifoRNN;
    }
    {
        YAML::Node poetizeShimoRNN;
        LFR_CONFIG_YAML_SETTER(poetizeShimoRNN, config.poetizeShimoRNN, model, model);
        yaml[lifuren::config::CONFIG_POETIZE_SHIMO_RNN] = poetizeShimoRNN;
    }
    {
        YAML::Node poetizeShiguiRNN;
        LFR_CONFIG_YAML_SETTER(poetizeShiguiRNN, config.poetizeShiguiRNN, model, model);
        yaml[lifuren::config::CONFIG_POETIZE_SHIGUI_RNN] = poetizeShiguiRNN;
    }
    {
        YAML::Node poetizeShixianRNN;
        LFR_CONFIG_YAML_SETTER(poetizeShixianRNN, config.poetizeShixianRNN, model, model);
        yaml[lifuren::config::CONFIG_POETIZE_SHIXIAN_RNN] = poetizeShixianRNN;
    }
    {
        YAML::Node poetizeShishengRNN;
        LFR_CONFIG_YAML_SETTER(poetizeShishengRNN, config.poetizeShishengRNN, model, model);
        yaml[lifuren::config::CONFIG_POETIZE_SHISHENG_RNN] = poetizeShishengRNN;
    }
    {
        YAML::Node poetizeLiduRNN;
        LFR_CONFIG_YAML_SETTER(poetizeLiduRNN, config.poetizeLiduRNN, model, model);
        yaml[lifuren::config::CONFIG_POETIZE_LIDU_RNN] = poetizeLiduRNN;
    }
    {
        YAML::Node poetizeSuxinRNN;
        LFR_CONFIG_YAML_SETTER(poetizeSuxinRNN, config.poetizeSuxinRNN, model, model);
        yaml[lifuren::config::CONFIG_POETIZE_SUXIN_RNN] = poetizeSuxinRNN;
    }
    {
        YAML::Node poetizeWanyueRNN;
        LFR_CONFIG_YAML_SETTER(poetizeWanyueRNN, config.poetizeWanyueRNN, model, model);
        yaml[lifuren::config::CONFIG_POETIZE_WANYUE_RNN] = poetizeWanyueRNN;
    }
    {
        YAML::Node paintCycleGAN;
        LFR_CONFIG_YAML_SETTER(paintCycleGAN, config.paintCycleGAN, model, model);
        yaml[lifuren::config::CONFIG_PAINT_CYCLE_GAN] = paintCycleGAN;
    }
    {
        YAML::Node paintSytleGAN;
        LFR_CONFIG_YAML_SETTER(paintSytleGAN, config.paintSytleGAN, model, model);
        yaml[lifuren::config::CONFIG_PAINT_STYLE_GAN] = paintSytleGAN;
    }
    {
        YAML::Node stableDiffusionCPP;
        LFR_CONFIG_YAML_SETTER(stableDiffusionCPP, config.stableDiffusionCPP, model,   model);
        LFR_CONFIG_YAML_SETTER(stableDiffusionCPP, config.stableDiffusionCPP, options, options);
        yaml[lifuren::config::CONFIG_STABLE_DIFFUSION_CPP] = stableDiffusionCPP;
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

bool lifuren::config::MarkConfig::operator==(const std::string& path) const {
    return this->path == path;
}
