#include "lifuren/Config.hpp"

#include <list>
#include <mutex>
#include <chrono>
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
        const auto& vk = v.first.template as<std::string>();        \
        const auto& vv = v.second;                                  \
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

std::string lifuren::config::base_dir       = "";

const std::string lifuren::config::CONFIG_CONFIG         = "config";
const std::string lifuren::config::CONFIG_AUDIO          = "audio";
const std::string lifuren::config::CONFIG_VIDEO          = "video";
const std::string lifuren::config::CONFIG_AUDIO_SHIKUANG = "audio-shikuang";
const std::string lifuren::config::CONFIG_VIDEO_WUDAOZI  = "video-wudaozi";

lifuren::config::Config lifuren::config::CONFIG{};

/**
 * 加载配置
 */
static void loadYaml(
    lifuren::config::Config& config, // 配置
    const std::string& name, // 配置名称
    const YAML::Node & yaml  // 配置内容
);
/**
 * @return YAML
 */
static YAML::Node toYaml();

std::string lifuren::config::Config::toYaml() {
    YAML::Node node = ::toYaml();
    std::stringstream stream;
    stream << node;
    return stream.str();
}

void loadYaml(lifuren::config::Config& config, const std::string& name, const YAML::Node& yaml) {
    if(lifuren::config::CONFIG_CONFIG == name) {
        const auto& tmp = yaml["tmp"];
        if(tmp) {
            config.tmp = tmp.as<std::string>();
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
    } else if(lifuren::config::CONFIG_VIDEO == name) {
        LFR_CONFIG_YAML_GETTER(config.video, yaml, path,   path,   std::string);
        LFR_CONFIG_YAML_GETTER(config.video, yaml, model,  model,  std::string);
        LFR_CONFIG_YAML_GETTER(config.video, yaml, client, client, std::string);
        const YAML::Node& clients = yaml["clients"];
        if(clients) {
            std::for_each(clients.begin(), clients.end(), [&config](const auto& client) {
                config.video.clients.emplace(client.template as<std::string>());
            });
        }
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
        YAML::Node video;
        LFR_CONFIG_YAML_SETTER(video, config.video, path,   path);
        LFR_CONFIG_YAML_SETTER(video, config.video, model,  model);
        LFR_CONFIG_YAML_SETTER(video, config.video, client, client);
        YAML::Node clients;
        std::for_each(config.video.clients.begin(), config.video.clients.end(), [&clients](auto& v) {
            clients.push_back(v);
        });
        video["clients"] = clients;
        yaml[lifuren::config::CONFIG_VIDEO] = video;
    }
    return yaml;
}

lifuren::config::Config lifuren::config::Config::loadFile() {
    const std::string path = lifuren::config::baseFile(lifuren::config::CONFIG_PATH);
    SPDLOG_DEBUG("加载配置文件：{}", path);
    lifuren::config::Config config{};
    YAML::Node yaml = lifuren::yaml::loadFile(path);
    if(!yaml || yaml.IsNull() || yaml.size() == 0) {
        return config;
    }
    for(auto iterator = yaml.begin(); iterator != yaml.end(); ++iterator) {
        const auto& key   = iterator->first.as<std::string>();
        const auto& value = iterator->second;
        try {
            loadYaml(config, key, value);
        } catch(...) {
            SPDLOG_ERROR("加载配置异常：{}", key);
        }
    }
    return config;
}

bool lifuren::config::Config::saveFile() {
    const std::string path = lifuren::config::baseFile(lifuren::config::CONFIG_PATH);
    SPDLOG_INFO("保存配置文件：{}", path);
    return lifuren::yaml::saveFile(::toYaml(), path);
}

void lifuren::config::init(const int argc, const char* const argv[]) {
    if(argc > 0) {
        lifuren::config::base_dir = std::filesystem::absolute(std::filesystem::path(argv[0]).parent_path()).string();
    }
    SPDLOG_DEBUG("执行目录：{}", lifuren::config::base_dir);
    lifuren::config::loadConfig();
}

std::string lifuren::config::baseFile(const std::string& path) {
    return lifuren::file::join({lifuren::config::base_dir, path}).string();
}

void lifuren::config::loadConfig() noexcept(true) {
    lifuren::config::CONFIG = lifuren::config::Config::loadFile();
}

size_t lifuren::config::uuid() noexcept(true) {
    static std::mutex mutex;
          static int index     = 0;
    const static int MIN_INDEX = 0;
    const static int MAX_INDEX = 100000;
    auto timePoint = std::chrono::system_clock::now();
    auto timestamp = std::chrono::system_clock::to_time_t(timePoint);
    auto localtime = std::localtime(&timestamp);
    int i = 0;
    {
        std::lock_guard<std::mutex> lock(mutex);
        i = index;
        if(++index >= MAX_INDEX) {
            index = MIN_INDEX;
        }
    }
    size_t id = 1000000000000000LL * (localtime->tm_year - 100) + // + 1900 - 2000
                10000000000000LL   * (localtime->tm_mon  +   1) +
                100000000000LL     *  localtime->tm_mday        +
                1000000000LL       *  localtime->tm_hour        +
                10000000LL         *  localtime->tm_min         +
                100000LL           *  localtime->tm_sec         +
                i;
    return id;
}
