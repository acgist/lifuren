/**
 * 文件工具
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_FILES_HPP
#define LFR_HEADER_CORE_FILES_HPP

#include <string>
#include <vector>
#include <filesystem>
#include <functional>

namespace lifuren {
namespace files   {

/**
 * 遍历文件列表
 * 
 * @param vector 列表
 * @param path   路径
 */
extern void listFiles(std::vector<std::string>& vector, const std::string& path);

/**
 * 遍历文件列表
 * 
 * @param vector 列表
 * @param path   路径
 * @param exts   文件后缀
 */
extern void listFiles(std::vector<std::string>& vector, const std::string& path, const std::vector<std::string>& exts);

/**
 * 遍历文件列表
 * 
 * @param vector    列表
 * @param path      路径
 * @param predicate 路径匹配
 */
extern void listFiles(std::vector<std::string>& vector, const std::string& path, const std::function<bool(const std::string& path)>& predicate);

/**
 * @param path 文件路径
 * 
 * @return 文本内容
 */
extern std::string loadFile(const std::string& path);

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
 * @param path 文件路径
 * 
 * @return 文件后缀
 */
extern std::string fileType(const std::string& path);

}
}

#endif // LFR_HEADER_CORE_FILES_HPP
