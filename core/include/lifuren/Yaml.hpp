/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * YAML工具
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_YAML_HPP
#define LFR_HEADER_CORE_YAML_HPP

#include <string>

// 枚举转换
#ifndef LFR_FORMAT_YAML_ENUM
#define LFR_FORMAT_YAML_ENUM(enumName, aValue, zValue, defaultValue)       \
template<>                                                                 \
struct YAML::convert<lifuren::enumName> {                                  \
    static Node encode(const lifuren::enumName& value) {                   \
        return YAML::Node(static_cast<int>(value));                        \
    }                                                                      \
    static bool decode(const YAML::Node& node, lifuren::enumName& value) { \
        const int v = node.as<int>();                                      \
        for(                                                               \
            int index  = static_cast<int>(lifuren::enumName::aValue);      \
            index     <= static_cast<int>(lifuren::enumName::zValue);      \
            ++index                                                        \
        ) {                                                                \
            if (v == index) {                                              \
                value = static_cast<lifuren::enumName>(v);                 \
                return true;                                               \
            }                                                              \
        }                                                                  \
        value = lifuren::enumName::defaultValue;                           \
        return true;                                                       \
    }                                                                      \
};
#endif

namespace YAML {

class Node;

} // END OF YAML

namespace lifuren::yaml {

/**
 * 加载配置文件
 * 
 * @return YAML
 */
extern YAML::Node loadFile(
    const std::string& path // 文件路径
);

/**
 * 保存配置文件
 * 
 * @return 是否成功
 */
extern bool saveFile(
    const YAML::Node & yaml, // YAML
    const std::string& path  // 文件路径
);

} // END OF lifuren::yaml

#endif // LFR_HEADER_CORE_YAML_HPP
