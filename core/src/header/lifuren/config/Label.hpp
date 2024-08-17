/**
 * 标签
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_CONFIG_LABEL_HPP
#define LFR_HEADER_CORE_CONFIG_LABEL_HPP

#include <map>
#include <string>
#include <vector>
#include <cstdint>

namespace lifuren {

// 音频标签路径
const char* const LABEL_AUDIO_PATH  = "../config/audio.yml";
// 图片标签路径
const char* const LABEL_IMAGE_PATH  = "../config/image.yml";
// 视频标签路径
const char* const LABEL_VIDEO_PATH  = "../config/video.yml";
// 诗词标签路径
const char* const LABEL_POETRY_PATH = "../config/poetry.yml";

class LabelFile;
class LabelText;

extern std::map<std::string, std::vector<LabelFile>> LABEL_AUDIO;
extern std::map<std::string, std::vector<LabelFile>> LABEL_IMAGE;
extern std::map<std::string, std::vector<LabelFile>> LABEL_VIDEO;
extern std::map<std::string, LabelText> LABEL_POETRY;

/**
 * 标签
 */
class Label {

public:
    Label();
    Label(const std::string& name, const std::string& alias);
    virtual ~Label();

public:
    // 标签名称
    std::string name;
    // 标签别名
    std::string alias;

public:
    /**
     * @return YAML
     */
    virtual std::string toYaml() = 0;

};

/**
 * 文件标签
 */
class LabelFile : public Label {

public:
    // 标签数组
    std::vector<std::string> labels;

public:
    LabelFile();
    /**
     * @param name   名称
     * @param labels 标签数组
     */
    LabelFile(const std::string& name);

public:
    virtual std::string toYaml() override;
    /**
     * @param path 文件路径
     * 
     * @return 映射
     */
    static std::map<std::string, std::vector<LabelFile>> loadFile(const std::string& path);

};

/**
 * 分词标签
 */
class LabelText : public Label {

public:
    // 韵律：题材、词牌
    std::string rhythmic;
    // 示例
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
    LabelText();
    /**
     * @param name 名称
     */
    LabelText(const std::string& name);

public:
    virtual std::string toYaml() override;
    /**
     * @param path 文件路径
     * 
     * @return 映射
     */
    static std::map<std::string, LabelText> loadFile(const std::string& path);

};

}

#endif // LFR_HEADER_CORE_CONFIG_LABEL_HPP
