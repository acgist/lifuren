/**
 * 文件工具
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_FILE_HPP
#define LFR_HEADER_CORE_FILE_HPP

#include <string>
#include <vector>
#include <filesystem>
#include <functional>
#include <initializer_list>

namespace lifuren {
namespace file    {

/**
 * @param path 文件路径
 * 
 * @return 文件是否存在
 */
inline bool exists(const std::string& path) {
    return std::filesystem::exists(std::filesystem::u8path(path));
}

/**
 * @param path 文件路径
 * 
 * @return 是否是文件
 */
inline bool isFile(const std::string& path) {
    return std::filesystem::is_regular_file(std::filesystem::u8path(path));
}

/**
 * @param path 文件路径
 * 
 * @return 是否是目录
 */
inline bool isDirectory(const std::string& path) {
    return std::filesystem::is_directory(std::filesystem::u8path(path));
}

/**
 * @param path 文件路径
 * 
 * @return 上级文件路径
 */
inline std::string parent(const std::string& path) {
    return std::filesystem::u8path(path).parent_path().string();
}

/**
 * @param list 文件路径列表
 * 
 * @return 文件路径
 */
inline std::filesystem::path join(std::initializer_list<std::string> list) {
    if(list.size() <= 0) {
        return {};
    }
    auto iterator = list.begin();
    std::filesystem::path path{ std::filesystem::u8path(*iterator) };
    ++iterator;
    for(; iterator != list.end(); ++iterator) {
        path /= std::filesystem::u8path(*iterator);
    }
    return path;
}

/**
 * 遍历文件列表
 * 
 * @param vector 列表
 * @param path   路径
 */
extern void listFile(std::vector<std::string>& vector, const std::string& path);

/**
 * 遍历文件列表
 * 
 * @param vector 列表
 * @param path   路径
 * @param exts   文件后缀
 */
extern void listFile(std::vector<std::string>& vector, const std::string& path, const std::vector<std::string>& exts);

/**
 * 遍历文件列表
 * 
 * @param vector    列表
 * @param path      路径
 * @param predicate 路径匹配
 */
extern void listFile(std::vector<std::string>& vector, const std::string& path, const std::function<bool(const std::string& path)>& predicate);

/**
 * @param path 文件路径
 * 
 * @return 文本内容
 */
extern std::string loadFile(const std::string& path);

/**
 * 读取文件
 * 数据使用完后调用`delete[]`释放资源
 * 
 * @param file   文件
 * @param data   数据
 * @param length 长度
 */
extern void loadFile(const std::string& path, char** data, size_t& length);

/**
 * @param path  文件路径
 * @param value 文件内容
 * 
 * @return 是否成功
 */
extern bool saveFile(const std::string& path, const std::string& value);

/**
 * @param path 文件路径
 * 
 * @return 是否成功
 */
extern bool createParent(const std::string& path);

/**
 * @param path 目录路径
 * 
 * @return 是否成功
 */
extern bool createFolder(const std::string& path);

/**
 * @parma path 目录路径
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
 * @param path 文件路径
 * 
 * @return 文件后缀
 */
extern std::string fileType(const std::string& path);

} // END OF file
} // END OF lifuren

#endif // LFR_HEADER_CORE_FILE_HPP
