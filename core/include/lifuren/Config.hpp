/**
 * 系统设置
 * 
 * @author acgist
 * 
 * TODO:
 * 1. 内容错误
 * 2. 诗经
 * 3. 元曲
 * 4. 五代诗词
 */
#ifndef LFR_HEADER_CORE_CONFIG_HPP
#define LFR_HEADER_CORE_CONFIG_HPP

#include <map>
#include <set>
#include <list>
#include <string>
#include <vector>

namespace lifuren::config {

class Config;
class Rhythm;

// 配置路径
#ifdef _WIN32
const char* const CONFIG_PATH = "../config/config-win.yml";
#else
const char* const CONFIG_PATH = "../config/config.yml";
#endif
// 格律路径
const char* const RHYTHM_PATH = "../config/rhythm.yml";

extern std::string base;           // 执行绝对路径
extern std::string httpServerHost; // 监听地址
extern int         httpServerPort; // 监听端口

const std::string LIFUREN_HIDDEN_FILE  = ".lifuren";        // 隐藏文件
const std::string MARK_MODEL_FILE      = "mark.model";      // 标记文件
const std::string INDEXDB_MODEL_FILE   = "indexDB.model";   // 向量文件
const std::string MAPPING_MODEL_FILE   = "mapping.model";   // 映射文件
const std::string EMBEDDING_MODEL_FILE = "embedding.model"; // 嵌入文件

// 全局配置名称
extern const std::string CONFIG_CONFIG;
extern const std::string CONFIG_HTTP_SERVER;
extern const std::string CONFIG_AUDIO;
extern const std::string CONFIG_IMAGE;
extern const std::string CONFIG_VIDEO;
extern const std::string CONFIG_POETRY;
extern const std::string CONFIG_MARK;
extern const std::string CONFIG_RAG;
extern const std::string CONFIG_EMBEDDING;
extern const std::string CONFIG_OLLAMA;
extern const std::string CONFIG_PEPPER;
extern const std::string CONFIG_ELASTICSEARCH;
extern const std::string CONFIG_ACT_GUANHANQIN;
extern const std::string CONFIG_ACT_TANGXIANZU;
extern const std::string CONFIG_PAINT_CYCLE_GAN;
extern const std::string CONFIG_PAINT_STYLE_GAN;
extern const std::string CONFIG_POETIZE_LIDU;
extern const std::string CONFIG_POETIZE_SUXIN;
extern const std::string CONFIG_COMPOSE_SHIKUANG;
extern const std::string CONFIG_COMPOSE_LIGUINIAN;

// 全局配置：config.yml
extern lifuren::config::Config CONFIG;
// 全局格律：rhythm.yml
extern std::map<std::string, Rhythm> RHYTHM;

/**
 * REST配置
 */
struct RestConfig {

    // 接口地址
    std::string api;
    // 账号
    std::string username;
    // 密码
    std::string password;
    // 授权方式
    std::string authType;
    // 授权地址
    std::string authPath;

};

/**
 * 模型配置
 */
struct ModelConfig {

    // 模型文件
    std::string model;

};

/**
 * 词嵌入终端配置
 */
struct EmbeddingClientConfig {

    // 请求地址
    std::string path;
    // 词嵌入模型
    std::string model;
    // 其他配置
    std::map<std::string, std::string> options{};

};

/**
 * 音频页面配置
 */
struct AudioConfig {

    // 终端名称
    std::string client;
    // 输出目录
    std::string output;
    // 终端列表
    std::set<std::string> clients;

};

/**
 * 图片页面配置
 */
struct ImageConfig {

    // 终端名称
    std::string client;
    // 输出目录
    std::string output;
    // 终端列表
    std::set<std::string> clients;

};

/**
 * 视频页面配置
 */
struct VideoConfig {

    // 终端名称
    std::string client;
    // 输出目录
    std::string output;
    // 终端列表
    std::set<std::string> clients;

};

/**
 * 诗词页面配置
 */
struct PoetryConfig {

    // 终端名称
    std::string client;
    // 终端列表
    std::set<std::string> clients;

};

/**
 * 标记页面配置
 */
struct MarkConfig {

    // 目录
    std::string path;

    // 路径相同即为相等
    bool operator==(const std::string& path) const;

};

/**
 * RAG配置
 */
struct RAGConfig {

    // RAG类型
    std::string type;
    // 返回数量
    size_t size;

};

/**
 * 词嵌入配置
 */
struct EmbeddingConfig {

    // 词嵌入类型
    std::string type;
    // 分词类型
    std::string participle;

};

/**
 * Ollama配置
 */
struct OllamaConfig : RestConfig {

    // 词嵌入终端
    EmbeddingClientConfig embeddingClient;

};

/**
 * pepper配置
 */
struct PepperConfig {

    // 文件路径
    std::string path;

};

/**
 * ElasticSearch配置
 */
struct ElasticSearchConfig : RestConfig {

};

/**
 * 通用设置
 */
class Config {

public:
    // 基础配置
    std::string tmp;
    // 复合配置
    lifuren::config::AudioConfig  audio {};
    lifuren::config::ImageConfig  image {};
    lifuren::config::VideoConfig  video {};
    lifuren::config::PoetryConfig poetry{};
    std::list<lifuren::config::MarkConfig> mark{};
    lifuren::config::RAGConfig           rag          {};
    lifuren::config::EmbeddingConfig     embedding    {};
    lifuren::config::OllamaConfig        ollama       {};
    lifuren::config::PepperConfig        pepper       {};
    lifuren::config::ElasticSearchConfig elasticsearch{};
    lifuren::config::ModelConfig actGuanhanqin   {};
    lifuren::config::ModelConfig actTangxianzu   {};
    lifuren::config::ModelConfig paintCycleGAN   {};
    lifuren::config::ModelConfig paintSytleGAN   {};
    lifuren::config::ModelConfig composeShikuang {};
    lifuren::config::ModelConfig composeLiguinian{};
    lifuren::config::ModelConfig poetizeLidu     {};
    lifuren::config::ModelConfig poetizeSuxin    {};

public:
    Config();
    virtual ~Config();

public:
    /**
     * @return YAML
     */
    std::string toYaml();

};

/**
 * 格律
 */
class Rhythm {

public:
    Rhythm();
    Rhythm(const std::string& rhythm);
    virtual ~Rhythm();

public:
    // 韵律：题材、词牌
    std::string rhythm;
    // 别名
    std::vector<std::string> alias;
    // 标题
    std::string title;
    // 内容
    std::string example;
    // 字数
    int fontSize = 0;
    // 段数
    int segmentSize = 0;
    // 分段规则
    std::vector<uint32_t> segmentRule;
    // 分词规则
    std::vector<uint32_t> participleRule;

public:
    /**
     * @return YAML
     */
    virtual std::string toYaml();
    /**
     * @param path 文件路径
     * 
     * @return 映射
     */
    static std::map<std::string, Rhythm> loadFile(const std::string& path);

};

/**
 * 加载配置
 * 
 * @return 配置
 */
extern lifuren::config::Config loadFile();

/**
 * @param path 文件路径
 * 
 * @return 配置
 */
extern lifuren::config::Config loadFile(const std::string& path);

/**
 * 保存配置
 * 
 * @return 是否成功
 */
extern bool saveFile();

/**
 * @param path 文件路径
 * 
 * @return 是否成功
 */
extern bool saveFile(const std::string& path);

/**
 * @param argc 参数长度
 * @param argv 参数内容
 */
extern void init(const int argc, const char* const argv[]);

/**
 * @param path 相对目录
 * 
 * @return 绝对路径
 */
extern std::string baseFile(const std::string& path);

/**
 * 加载配置
 * 
 * @see #CONFIG_PATH
 * @see #RHYTHM_PATH
 */
extern void loadConfig() noexcept;

} // END OF lifuren::config

#endif // LFR_HEADER_CORE_CONFIG_HPP
