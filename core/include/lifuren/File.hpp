/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 文件
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
 * @param path 文件路径
 * 
 * @return 文件是否存在
 */
inline bool exists(const std::string& path) {
    return std::filesystem::exists(std::filesystem::path(path));
}

/**
 * @param path 文件路径
 * 
 * @return 是否是文件
 */
inline bool is_file(const std::string& path) {
    return std::filesystem::is_regular_file(std::filesystem::path(path));
}

/**
 * @param path 文件路径
 * 
 * @return 是否是目录
 */
inline bool is_folder(const std::string& path) {
    return std::filesystem::is_directory(std::filesystem::path(path));
}

/**
 * @param path 文件路径
 * 
 * @return 是否是目录
 */
inline bool is_directory(const std::string& path) {
    return lifuren::file::is_folder(path);
}

/**
 * @param path 文件路径
 * 
 * @return 上级文件路径
 */
inline std::string parent(const std::string& path) {
    return std::filesystem::path(path).parent_path().string();
}

/**
 * @param path 目录路径
 * 
 * @return 是否成功
 */
inline bool createFolder(const std::filesystem::path& path) {
    if(std::filesystem::exists(path)) {
        return true;
    }
    return std::filesystem::create_directories(path);
}

/**
 * @param path 目录路径
 * 
 * @return 是否成功
 */
inline bool createFolder(const std::string& path) {
    return lifuren::file::createFolder(std::filesystem::path(path));
}

/**
 * @param path 目录路径
 * 
 * @return 是否成功
 */
inline bool createDirectory(const std::filesystem::path& path) {
    return lifuren::file::createFolder(path);
}

/**
 * @param path 目录路径
 * 
 * @return 是否成功
 */
inline bool createDirectory(const std::string& path) {
    return lifuren::file::createFolder(path);
}

/**
 * @param path 目录路径
 * 
 * @return 是否成功
 */
inline bool createParent(const std::filesystem::path& path) {
    auto parent = path.parent_path();
    if(std::filesystem::exists(parent)) {
        return true;
    }
    return std::filesystem::create_directories(parent);
}

/**
 * @param path 文件路径
 * 
 * @return 是否成功
 */
inline bool createParent(const std::string& path) {
    return lifuren::file::createParent(std::filesystem::path(path));
}

/**
 * @param path 文件路径
 * 
 * @return 文件大小
 */
inline size_t file_size(const std::filesystem::path& path) {
    return std::filesystem::file_size(path);
}

/**
 * @param path 文件路径
 * 
 * @return 文件大小
 */
inline size_t file_size(const std::string& path) {
    return lifuren::file::file_size(std::filesystem::path(path));
}

/**
 * @param list 文件路径列表
 * 
 * @return 文件绝对路径
 */
inline std::filesystem::path join(std::initializer_list<std::string> list) {
    if(list.size() <= 0) {
        return {};
    }
    std::filesystem::path path{};
    for(auto iterator = list.begin(); iterator != list.end(); ++iterator) {
        if(iterator->empty()) {
            continue;
        }
        path /= std::filesystem::path(*iterator);
    }
    return std::filesystem::absolute(path);
}

/**
 * @param file   文件路径
 * @param suffix 新的后缀
 * @param prefix 新的前缀
 * 
 * @return 文件路径
 */
inline std::string modify_filename(const std::string& file, const std::string& suffix, const std::string& prefix = "") {
    if(file.empty()) {
        return {};
    }
    const auto pos = file.find_last_of('.');
    if(pos == std::string::npos) {
        return {};
    }
    if(prefix.empty()) {
        return file.substr(0, pos) + suffix;
    } else {
        return file.substr(0, pos) + "_" + prefix + suffix;
    }
}

/**
 * 遍历文件列表
 * 
 * @param vector 文件列表
 * @param path   文件路径
 */
extern void list_file(std::vector<std::string>& vector, const std::string& path);

/**
 * 遍历文件列表
 * 
 * @param vector 文件列表
 * @param path   文件路径
 * @param suffix 文件后缀
 */
extern void list_file(std::vector<std::string>& vector, const std::string& path, const std::vector<std::string>& suffix);

/**
 * 遍历文件列表
 * 
 * @param vector    文件列表
 * @param path      文件路径
 * @param predicate 文件匹配：是否匹配成功(文件路径)
 */
extern void list_file(std::vector<std::string>& vector, const std::string& path, const std::function<bool(const std::string& path)>& predicate);

} // END OF lifuren::file

namespace lifuren::string {

static const char* const EMPTY_CHARS = " \t\r\n"; // 空白字符

/**
 * @param value 字符串
 */
inline void toLower(std::string& value) {
    #if _WIN32
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    #else
    std::transform(value.begin(), value.end(), value.begin(), [](const char& v) -> char {
        return std::tolower(v);
    });
    #endif
}

/**
 * @param value 字符串
 * 
 * @return 去掉空格后的字符串
 */
inline std::string trim(const std::string& value) {
    std::size_t index = value.find_first_not_of(EMPTY_CHARS);
    if(index == std::string::npos) {
        return {};
    }
    std::size_t jndex = value.find_last_not_of(EMPTY_CHARS);
    return value.substr(index, jndex + 1 - index);
}
    
} // END OF lifuren::string

#endif // LFR_HEADER_CORE_FILE_HPP
