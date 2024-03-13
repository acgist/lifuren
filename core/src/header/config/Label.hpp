/**
 * 标签
 * 描述训练数据
 * 
 * @author acgist
 */
#pragma once

#include <map>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#include "../Logger.hpp"
#include "../utils/Files.hpp"

namespace lifuren {

// 音频标签路径
const char* const LABEL_AUDIO_PATH  = "../config/audio.json";
// 图片标签路径
const char* const LABEL_IMAGE_PATH  = "../config/image.json";
// 视频标签路径
const char* const LABEL_VIDEO_PATH  = "../config/video.json";
// 诗词标签路径
const char* const LABEL_POETRY_PATH = "../config/poetry.json";

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
    // 标签名称
    std::string name;
    // 标签别名
    std::string alias;
    // JSON序列化
    // NLOHMANN_DEFINE_TYPE_INTRUSIVE(Label, name, alias);
    // NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Label, name, alias);
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Label, name, alias);
    // NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Label, name, alias);

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON();

};

/**
 * 文件标签
 */
class LabelFile : public Label {

public:
    // 标签数组
    std::vector<std::string> labels;
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(LabelFile, name, alias, labels);

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON() override;
    /**
     * @param path 文件路径
     * 
     * @return 文件内容
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
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(LabelText, name, alias, example, fontSize, segmentSize, segmentRule, participleRule);

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON() override;
    /**
     * @param path 文件路径
     * 
     * @return 文件内容
     */
    static std::map<std::string, LabelText> loadFile(const std::string& path);

};

}
