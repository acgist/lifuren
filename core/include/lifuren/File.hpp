/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 文件工具
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_FILE_HPP
#define LFR_HEADER_CORE_FILE_HPP

#include <string>
#include <vector>
#include <filesystem>
#include <functional>
#include <initializer_list>

namespace lifuren::file {

/**
 * @return 文件是否存在
 */
inline bool exists(
    const std::string& path // 文件路径
) {
    return std::filesystem::exists(std::filesystem::path(path));
}

/**
 * @return 是否是文件
 */
inline bool is_file(
    const std::string& path // 文件路径
) {
    return std::filesystem::is_regular_file(std::filesystem::path(path));
}

/**
 * @return 是否是目录
 */
inline bool is_folder(
    const std::string& path // 文件路径
) {
    return std::filesystem::is_directory(std::filesystem::path(path));
}

/**
 * @return 是否是目录
 */
inline bool is_directory(
    const std::string& path // 文件路径
) {
    return is_folder(path);
}

/**
 * @return 上级文件路径
 */
inline std::string parent(
    const std::string& path // 文件路径
) {
    return std::filesystem::path(path).parent_path().string();
}

/**
 * @return 文件路径
 */
inline std::filesystem::path join(
    std::initializer_list<std::string> list // 文件路径列表
) {
    if(list.size() <= 0) {
        return {};
    }
    std::filesystem::path path{};
    for(auto iterator = list.begin(); iterator != list.end(); ++iterator) {
        path /= std::filesystem::path(*iterator);
    }
    return path;
}

/**
 * 遍历文件列表
 */
extern void listFile(
    std::vector<std::string>& vector, // 文件列表
    const std::string       & path    // 路径
);

/**
 * 遍历文件列表
 */
extern void listFile(
    std::vector<std::string>      & vector, // 文件列表
    const std::string             & path,   // 路径
    const std::vector<std::string>& suffix  // 文件后缀
);

/**
 * 遍历文件列表
 */
extern void listFile(
    std::vector<std::string>& vector, // 文件列表
    const std::string       & path,   // 路径
    const std::function<bool(const std::string& path)>& predicate // 路径匹配
);

/**
 * @return 文本内容
 */
extern std::string loadFile(
    const std::string& path // 文件路径
);

/**
 * @return 文件内容
 */
extern std::vector<char> loadBlobFile(
    const std::string& path // 文件路径
);

/**
 * @return 是否成功
 */
extern bool saveFile(
    const std::string& path, // 文件路径
    const std::string& value // 文件内容
);

/**
 * @return 是否成功
 */
inline bool createFolder(
    const std::filesystem::path& path // 目录路径
) {
    if(std::filesystem::exists(path)) {
        return true;
    }
    return std::filesystem::create_directories(path);
}

/**
 * @return 是否成功
 */
inline bool createFolder(
    const std::string& path // 目录路径
) {
    return createFolder(std::filesystem::path(path));
}

/**
 * @return 是否成功
 */
inline bool createDirectory(
    const std::filesystem::path& path // 目录路径
) {
    return createFolder(path);
}

/**
 * @return 是否成功
 */
inline bool createDirectory(
    const std::string& path // 目录路径
) {
    return createFolder(path);
}

/**
 * @return 是否成功
 */
inline bool createParent(
    const std::filesystem::path& path // 目录路径
) {
    auto parent = path.parent_path();
    if(std::filesystem::exists(parent)) {
        return true;
    }
    return std::filesystem::create_directories(parent);
}

/**
 * @return 是否成功
 */
inline bool createParent(
    const std::string& path // 文件路径
) {
    return createParent(std::filesystem::path(path));
}

/**
 * @return 文件后缀：.cpp/.hpp/.zip
 */
extern std::string fileSuffix(
    const std::string& path // 文件路径
);

} // END OF lifuren::file

#endif // LFR_HEADER_CORE_FILE_HPP
