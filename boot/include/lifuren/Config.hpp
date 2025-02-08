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
class Rhythm;

// 配置路径
#ifdef _WIN32
const char* const CONFIG_PATH = "../config/config-win.yml";
#else
const char* const CONFIG_PATH = "../config/config.yml";
#endif

extern std::string base_dir;       // 启动路径：项目启动绝对路径
extern std::string restServerHost; // 监听地址
extern int         restServerPort; // 监听端口

const std::string LIFUREN_HIDDEN_FILE  = ".lifuren";        // 隐藏文件
const std::string EMBEDDING_MODEL_FILE = "embedding.model"; // 嵌入文件：训练嵌入数据集

const std::string DATASET_TRAIN = "train"; // 训练数据集
const std::string DATASET_VAL   = "val";   // 验证数据集
const std::string DATASET_TEST  = "test";  // 测试数据集

// 全局配置名称
extern const std::string CONFIG_CONFIG;
extern const std::string CONFIG_REST_SERVER;
extern const std::string CONFIG_AUDIO;
extern const std::string CONFIG_VIDEO;
extern const std::string CONFIG_AUDIO_SHIKUANG;
extern const std::string CONFIG_VIDEO_WUDAOZI;

// 全局配置：config.yml
extern lifuren::config::Config CONFIG;

/**
 * 注意：一秒钟的并发不能超过十万
 * 
 * @return ID(yyMMddHHmmss'xxxxx)
 */
extern size_t uuid() noexcept(true);

/**
 * REST配置
 */
struct RestConfig {

    std::string api;      // 接口地址
    std::string username; // 账号
    std::string password; // 密码
    std::string authType; // 授权方式
    std::string authPath; // 授权地址

};

/**
 * 模型配置
 */
struct ModelConfig {

    std::string path;   // 文件目录
    std::string model;  // 模型路径（默认为空=文件目录/.lifuren/model-name.pt）
    std::string client; // 终端名称
    std::set<std::string> clients; // 终端列表

};

/**
 * 音频配置
 */
struct AudioConfig : public ModelConfig {

};

/**
 * 视频配置
 */
struct VideoConfig : public ModelConfig {

};

/**
 * 通用设置
 */
class Config {

public:
    std::string tmp;
    lifuren::config::AudioConfig audio{};
    lifuren::config::VideoConfig video{};

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
 * @see #RHYTHM_PATH
 */
extern void loadConfig() noexcept(true);

} // END OF lifuren::config

#endif // LFR_HEADER_BOOT_CONFIG_HPP
