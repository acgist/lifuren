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

#include "Files.hpp"
#include "Logger.hpp"

#include "nlohmann/json.hpp"

namespace lifuren {

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
    // NLOHMANN_DEFINE_TYPE_INTRUSIVE(Label, name);
    // NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Label, name);
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Label, name);
    // NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Label, name);

public:
    Label();
    virtual ~Label();
    /**
     * @param json JSON
     */
    Label(const std::string& json);

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON();
    /**
     * @param path 配置路径
     */
    virtual void loadFile(const std::string& path);

};

/**
 * 文件标签
 */
class LabelFile : public Label {

public:
    // 标签数组
    std::vector<std::string> labels;
    // 下级标签
    std::map<std::string, LabelFile> children;
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(LabelFile, name, labels, children);

public:
    LabelFile();
    virtual ~LabelFile();
    /**
     * @param json JSON
     */
    LabelFile(const std::string& json);

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON() override;

};

/**
 * 分词标签
 */
class LabelText : public Label {

public:
    // 字数
    int fontSize = 0;
    // 段数
    int segmentSize = 0;
    // 分段规则
    std::vector<int> segmentRule;
    // 分词规则
    std::vector<int> participleRule;
    // JSON序列化
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(LabelText, name, fontSize, segmentSize, segmentRule, participleRule);

public:
    LabelText();
    virtual ~LabelText();
    /**
     * @param json JSON
     */
    LabelText(const std::string& json);

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON() override;

};

// 音频标签路径
const char* const LABEL_AUDIO_PATH  = "../config/audio.json";
// 图片标签路径
const char* const LABEL_IMAGE_PATH  = "../config/image.json";
// 视频标签路径
const char* const LABEL_VIDEO_PATH  = "../config/video.json";
// 诗词标签路径
const char* const LABEL_POETRY_PATH = "../config/poetry.json";

extern LabelFile LABEL_AUDIO;
extern LabelFile LABEL_IMAGE;
extern LabelFile LABEL_VIDEO;
extern LabelText LABEL_POETRY;

}
