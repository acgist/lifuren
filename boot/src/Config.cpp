#include "lifuren/Config.hpp"

#include <mutex>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <filesystem>

#include "spdlog/spdlog.h"

#include "yaml-cpp/yaml.h"

#include "lifuren/File.hpp"

// 读取配置
#ifndef LFR_CONFIG_YAML_GETTER
#define LFR_CONFIG_YAML_GETTER(config, yaml, key, name, type) \
const auto& name = yaml[#key];                                \
if(name && !name.IsNull() && name.IsScalar()) {               \
    config.name = name.template as<type>();                   \
}
#endif

std::string lifuren::config::base_dir = "";

lifuren::config::Config lifuren::config::CONFIG{ };

/**
 * @param config 配置
 * @param name   名称
 * @param yaml   内容
 */
static void loadYaml(lifuren::config::Config& config, const std::string& name, const YAML::Node & yaml);

/**
 * @return YAML
 */
static YAML::Node toYaml();

/**
 * @param path 文件路径
 * 
 * @return YAML
 */
static YAML::Node loadFile(const std::string& path);

/**
 * @param yaml YAML
 * @param path 文件路径
 * 
 * @return 是否成功
 */
static bool saveFile(const YAML::Node & yaml, const std::string& path);

static void loadYaml(lifuren::config::Config& config, const std::string& name, const YAML::Node& yaml) {
    if("config" == name) {
        LFR_CONFIG_YAML_GETTER(config, yaml, tmp,    tmp,    std::string);
        LFR_CONFIG_YAML_GETTER(config, yaml, output, output, std::string);
    } else if("model" == name) {
        LFR_CONFIG_YAML_GETTER(config, yaml, wudaozi,  model_wudaozi,  std::string);
        LFR_CONFIG_YAML_GETTER(config, yaml, shikuang, model_shikuang, std::string);
    } else {
        SPDLOG_DEBUG("没有适配加载配置类型：{}", name);
    }
}

static YAML::Node toYaml() {
    const auto& config = lifuren::config::CONFIG;
    YAML::Node yaml;
    {
        YAML::Node node;
        node["tmp"]    = lifuren::config::CONFIG.tmp;
        node["output"] = lifuren::config::CONFIG.output;
        yaml["config"] = node;
    }
    {
        YAML::Node node;
        node["wudaozi"]  = lifuren::config::CONFIG.model_wudaozi;
        node["shikuang"] = lifuren::config::CONFIG.model_shikuang;
        yaml["model"]    = node;
    }
    return yaml;
}

static YAML::Node loadFile(const std::string& path) {
    if(!lifuren::file::exists(path) || !lifuren::file::is_file(path)) {
        return {};
    }
    try {
        return YAML::LoadFile(path);
    } catch(const std::exception& e) {
        SPDLOG_ERROR("加载YAML异常：{}", e.what());
    }
    return {};
}

static bool saveFile(const YAML::Node& yaml, const std::string& path) {
    lifuren::file::createParent(path);
    std::ofstream output;
    output.open(path, std::ios_base::out | std::ios_base::trunc);
    if(!output.is_open()) {
        SPDLOG_WARN("打开配置文件失败：{}", path);
        return false;
    }
    output << yaml;
    output.close();
    return true;
}

lifuren::config::Config lifuren::config::Config::loadFile() {
    const std::string path = lifuren::config::baseFile(lifuren::config::CONFIG_PATH);
    SPDLOG_DEBUG("加载配置文件：{}", path);
    lifuren::config::Config config{ };
    YAML::Node yaml = ::loadFile(path);
    if(!yaml || yaml.IsNull() || yaml.size() == 0) {
        return config;
    }
    for(auto iterator = yaml.begin(); iterator != yaml.end(); ++iterator) {
        const auto& key   = iterator->first.as<std::string>();
        const auto& value = iterator->second;
        try {
            ::loadYaml(config, key, value);
        } catch(...) {
            SPDLOG_ERROR("加载配置文件异常：{}", key);
        }
    }
    return config;
}

bool lifuren::config::Config::saveFile() {
    const std::string path = lifuren::config::baseFile(lifuren::config::CONFIG_PATH);
    SPDLOG_INFO("保存配置文件：{}", path);
    return ::saveFile(::toYaml(), path);
}

void lifuren::config::init(const int argc, const char* const argv[]) {
    if(argc > 0) {
        lifuren::config::base_dir = std::filesystem::absolute(std::filesystem::path(argv[0]).parent_path()).string();
    }
    SPDLOG_DEBUG("项目启动绝对路径：{}", lifuren::config::base_dir);
    lifuren::config::CONFIG = lifuren::config::Config::loadFile();
}

std::string lifuren::config::baseFile(const std::string& path) {
    return lifuren::file::join({lifuren::config::base_dir, path}).string();
}

size_t lifuren::config::uuid() noexcept(true) {
    auto timePoint = std::chrono::system_clock::now();
    auto timestamp = std::chrono::system_clock::to_time_t(timePoint);
    auto localtime = std::localtime(&timestamp);
    int i = 0;
    {
              static int index     = 0;
        const static int MIN_INDEX = 0;
        const static int MAX_INDEX = 100000;
        static std::mutex mutex;
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
