/**
 * YAML工具
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_YAMLS_HPP
#define LFR_HEADER_CORE_YAMLS_HPP

#include <string>

#include "yaml-cpp/yaml.h"

// 枚举转换
#ifndef LFR_YAML_ENUM
#define LFR_YAML_ENUM(enumName, aValue, zValue, defaultValue)              \
template <>                                                                \
struct YAML::convert<lifuren::enumName> {                                  \
    static Node encode(const lifuren::enumName& value) {                   \
        return YAML::Node(static_cast<int>(value));                        \
    }                                                                      \
    static bool decode(const YAML::Node& node, lifuren::enumName& value) { \
        int v = node.as<int>();                                            \
        for(                                                               \
            int index  = static_cast<int>(lifuren::enumName::aValue);      \
            index     <= static_cast<int>(lifuren::enumName::zValue);      \
            ++index                                                        \
        ) {                                                                \
            if (v == index) {                                              \
                value = (lifuren::enumName) v;                             \
                return true;                                               \
            }                                                              \
        }                                                                  \
        value = lifuren::enumName::defaultValue;                           \
        return true;                                                       \
    }                                                                      \
};
#endif

namespace lifuren {
namespace yamls   {

/**
 * @param path 文件路径
 * 
 * @return YAML
 */
extern YAML::Node loadFile(const std::string& path);

/**
 * @param yaml YAML
 * @param path 文件路径
 */
extern bool saveFile(const YAML::Node& yaml, const std::string& path);

}
}

#endif // LFR_HEADER_CORE_YAMLS_HPP
