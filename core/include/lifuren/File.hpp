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
inline bool is_directory(const std::string& path) {
    return std::filesystem::is_directory(std::filesystem::path(path));
}

/**
 * @param path 目录路径
 * 
 * @return 是否成功
 */
inline bool create_parent_directory(const std::string& path) {
    auto parent = std::filesystem::path(path).parent_path();
    if(std::filesystem::exists(parent)) {
        return true;
    }
    return std::filesystem::create_directories(parent);
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
        return file + "_";
    }
    const auto pos = file.find_last_of('.');
    if(pos == std::string::npos) {
        return file + "_";
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
extern void list_file(std::vector<std::string>& vector, const std::string& path, const std::function<bool(const std::string&)>& predicate);

} // END OF lifuren::file

#endif // LFR_HEADER_CORE_FILE_HPP
