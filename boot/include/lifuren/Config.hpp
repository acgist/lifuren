/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 系统配置
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_BOOT_CONFIG_HPP
#define LFR_HEADER_BOOT_CONFIG_HPP

#include <map>
#include <set>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdint>

namespace lifuren::config {

/**
 * 模型参数
 */
struct ModelParams {

    float       lr         { 0.001F      }; // 学习率
    size_t      batch_size { 100         }; // 批量大小
    size_t      epoch_count{ 128         }; // 训练次数
    size_t      thread_size{ 0           }; // 线程数量
    size_t      class_size { 2           }; // 分类数量
    bool        classify   { false       }; // 分类任务
    bool        check_point{ false       }; // 保存快照
    std::string model_name { "lifuren"   }; // 模型名称
    std::string check_path { "./lifuren" }; // 快照路径
    std::string train_path { "./train"   }; // 训练数据集路径
    std::string val_path   { "./val"     }; // 验证数据集路径
    std::string test_path  { "./test"    }; // 测试数据集路径

};

class Config;

// 配置路径
#ifdef _WIN32
const char* const CONFIG_PATH = "../config/config-win.yml";
#else
const char* const CONFIG_PATH = "../config/config.yml";
#endif

extern std::string base_dir; // 启动路径：项目启动绝对路径

const std::string LIFUREN_HIDDEN_FILE  = ".lifuren";        // 隐藏文件
const std::string EMBEDDING_MODEL_FILE = "embedding.model"; // 嵌入文件：训练嵌入数据集

const std::string DATASET_TRAIN = "train"; // 训练数据集
const std::string DATASET_VAL   = "val";   // 验证数据集
const std::string DATASET_TEST  = "test";  // 测试数据集

// 全局配置：config.yml
extern lifuren::config::Config CONFIG;

/**
 * 注意：一秒钟的并发不能超过十万
 * 
 * @return ID(yyMMddHHmmss'xxxxx)
 */
extern size_t uuid() noexcept(true);

/**
 * 通用设置
 */
class Config {

public:
    std::string tmp;
    std::string output;
    std::string model_bach;
    std::string model_chopin;
    std::string model_mozart;
    std::string model_wudaozi;
    std::string model_shikuang;
    std::string model_beethoven;


public:
    /**
     * @return YAML
     */
    std::string toYaml();

public:
    /**
     * 加载配置
     * 
     * @return 配置
     */
    static lifuren::config::Config loadFile();

    /**
     * 保存配置
     * 
     * @return 是否成功
     */
    static bool saveFile();

};

/**
 * 初始化系统环境
 */
extern void init(
    const int         argc,  // 参数长度
    const char* const argv[] // 参数内容
);

/**
 * @return 绝对路径
 */
extern std::string baseFile(
    const std::string& path // 相对目录
);

/**
 * 加载配置
 * 
 * @see #CONFIG_PATH
 */
extern void loadConfig() noexcept(true);

} // END OF lifuren::config

#endif // LFR_HEADER_BOOT_CONFIG_HPP
