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
#ifndef LFR_HEADER_CORE_CONFIG_HPP
#define LFR_HEADER_CORE_CONFIG_HPP

#include <string>
#include <cstdlib>
#include <cstdint>

// 删除指针
#ifndef LFR_DELETE_PTR
#define LFR_DELETE_PTR(ptr) \
    if(ptr != nullptr) {    \
        delete ptr;         \
        ptr = nullptr;      \
    }
#endif

// 删除指针
#ifndef LFR_DELETE_THIS_PTR
#define LFR_DELETE_THIS_PTR(ptr) \
    if(this->ptr != nullptr) {   \
        delete this->ptr;        \
        this->ptr = nullptr;     \
    }
#endif

namespace lifuren::config {

/**
 * 模型参数
 */
struct ModelParams {

    float       lr         { 0.001F      }; // 学习率
    float       grad_clip  { 0.0F        }; // 梯度裁剪
    size_t      batch_size { 100         }; // 批量大小
    size_t      epoch_size { 128         }; // 训练轮次
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

const std::string DATASET_TRAIN = "train"; // 训练数据集
const std::string DATASET_VAL   = "val";   // 验证数据集
const std::string DATASET_TEST  = "test";  // 测试数据集

/**
 * 配置
 */
class Config {

public:
    std::string tmp;    // 临时目录
    std::string output; // 输出目录
    std::string model_wudaozi; // 视频生成模型文件

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
 * 注意：一秒钟的并发不能超过十万
 * 
 * @return ID(yyMMddHHmmss'xxxxx)
 */
extern size_t uuid() noexcept(true);

/**
 * @param path 相对路径
 * 
 * @return 绝对路径
 */
extern std::string baseFile(const std::string& path);

} // END OF lifuren::config

#endif // LFR_HEADER_CORE_CONFIG_HPP
