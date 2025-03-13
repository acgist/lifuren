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

// 配置写出
#ifndef LFR_CONFIG_YAML_SETTER
#define LFR_CONFIG_YAML_SETTER(target, source, name, key) \
target[#key] = source.name;
#endif

std::string lifuren::config::base_dir = "";

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
    if("config" == name) {
        LFR_CONFIG_YAML_GETTER(config, yaml, tmp,    tmp, std::string);
        LFR_CONFIG_YAML_GETTER(config, yaml, output, output, std::string);
    } else if("model" == name) {
        LFR_CONFIG_YAML_GETTER(config, yaml, bach,      model_bach,      std::string);
        LFR_CONFIG_YAML_GETTER(config, yaml, chopin,    model_chopin,    std::string);
        LFR_CONFIG_YAML_GETTER(config, yaml, mozart,    model_mozart,    std::string);
        LFR_CONFIG_YAML_GETTER(config, yaml, wudaozi,   model_wudaozi,   std::string);
        LFR_CONFIG_YAML_GETTER(config, yaml, shikuang,  model_shikuang,  std::string);
        LFR_CONFIG_YAML_GETTER(config, yaml, beethoven, model_beethoven, std::string);
    } else {
        SPDLOG_DEBUG("配置没有适配加载：{}", name);
    }
}

static YAML::Node toYaml() {
    const auto& config = lifuren::config::CONFIG;
    YAML::Node yaml;
    {
        YAML::Node config;
        config["tmp"]    = lifuren::config::CONFIG.tmp;
        config["output"] = lifuren::config::CONFIG.output;
        yaml["config"] = config;
    }
    {
        YAML::Node model;
        model["bach"]      = lifuren::config::CONFIG.model_bach;
        model["chopin"]    = lifuren::config::CONFIG.model_chopin;
        model["mozart"]    = lifuren::config::CONFIG.model_mozart;
        model["wudaozi"]   = lifuren::config::CONFIG.model_wudaozi;
        model["shikuang"]  = lifuren::config::CONFIG.model_shikuang;
        model["beethoven"] = lifuren::config::CONFIG.model_beethoven;
        yaml["model"] = model;
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
            // auto exptr = std::current_exception();
            // std::rethrow_exception(exptr);
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
