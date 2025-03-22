/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 配置
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_BOOT_CONFIG_HPP
#define LFR_HEADER_BOOT_CONFIG_HPP

#include <string>
#include <cstdlib>
#include <cstdint>

namespace lifuren::config {

/**
 * 模型训练参数
 */
struct ModelParams {

    float       lr         { 0.001F      }; // 学习率
    size_t      batch_size { 100         }; // 批量大小
    size_t      epoch_size { 128         }; // 训练次数
    size_t      thread_size{ 0           }; // 线程数量
    size_t      class_size { 2           }; // 任务分类数量
    bool        classify   { false       }; // 是否分类任务
    bool        check_point{ false       }; // 是否保存快照
    std::string model_name { "lifuren"   }; // 模型名称
    std::string model_path { "./lifuren" }; // 模型路径
    std::string train_path { "./train"   }; // 训练数据集路径
    std::string val_path   { "./val"     }; // 验证数据集路径
    std::string test_path  { "./test"    }; // 测试数据集路径

};

class Config;

#ifdef _WIN32
const char* const CONFIG_PATH = "../config/config-win.yml";
#else
const char* const CONFIG_PATH = "../config/config.yml";
#endif

extern std::string base_dir; // 项目启动绝对路径

extern lifuren::config::Config CONFIG; // 全局配置

const std::string LIFUREN_HIDDEN_FILE    = ".lifuren";        // 隐藏文件
const std::string LIFUREN_EMBEDDING_FILE = "model.embedding"; // 嵌入文件：训练嵌入数据集

const std::string DATASET_TRAIN = "train"; // 训练数据集
const std::string DATASET_VAL   = "val";   // 验证数据集
const std::string DATASET_TEST  = "test";  // 测试数据集

/**
 * 注意：一秒钟的并发不能超过十万
 * 
 * @return ID(yyMMddHHmmss'xxxxx)
 */
extern size_t uuid() noexcept(true);

/**
 * 配置
 */
class Config {

public:
    std::string tmp;            // 临时目录
    std::string output;         // 输出目录
    std::string model_bach;     // 巴赫模型文件
    std::string model_chopin;   // 肖邦模型文件
    std::string model_mozart;   // 莫扎特模型文件
    std::string model_shikuang; // 师旷模型文件

public:
    /**
     * @return YAML
     */
    std::string toYaml();

public:
    /**
     * @return 配置
     */
    static lifuren::config::Config loadFile();

    /**
     * @return 是否成功
     */
    static bool saveFile();

};

/**
 * @param argc 参数长度
 * @param argv 参数内容
 */
extern void init(const int argc, const char* const argv[]);

/**
 * @param path 相对路径
 * 
 * @return 绝对路径
 */
extern std::string baseFile(const std::string& path);

} // END OF lifuren::config

#endif // LFR_HEADER_BOOT_CONFIG_HPP
