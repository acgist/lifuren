/**
 * 文件工具
 * 
 * @author acgist
 */
#pragma once

#include <vector>
#include <algorithm>
#include <filesystem>

#include "Logger.hpp"
#include "Strings.hpp"

namespace lifuren {
namespace files {

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
 * @param exts   后缀
 */
extern void listFiles(std::vector<std::string>& vector, const std::string& path, const std::vector<std::string>& exts);

/**
 * 遍历文件列表
 * 
 * @param vector    列表
 * @param path      路径
 * @param predicate 路径匹配
 */
template <typename Predicate>
extern void listFiles(std::vector<std::string>& vector, const std::string& path, const Predicate& predicate);

}
}