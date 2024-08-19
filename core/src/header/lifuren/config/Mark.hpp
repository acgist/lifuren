/**
 * 标记
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_CONFIG_MARK_HPP
#define LFR_HEADER_CORE_CONFIG_MARK_HPP

#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace lifuren {

/**
 * 标记
 */
class Mark {

public:
    // 标签数组
    std::vector<std::string> labels;
    // JSON解析
    // NLOHMANN_DEFINE_TYPE_INTRUSIVE
    // NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Mark, labels);
    // NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT

public:
    Mark();
    virtual ~Mark();
    /**
     * @param json JSON
     */
    Mark(const std::string& json);

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON();

};

/**
 * 文件标记
 */
class MarkFile : public Mark {

public:
    // 本地文件
    std::string file;
    // JSON解析
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(MarkFile, file, labels);

public:
    MarkFile();
    virtual ~MarkFile();
    /**
     * @param json JSON
     */
    MarkFile(const std::string& json);

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON() override;

};

/**
 * 文本标记
 */
class MarkText : public Mark {

public:
    // 文本名称
    std::string name;
    // 文本内容
    std::string text;
    // 标记名称
    std::string label;
    // JSON解析
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(MarkText, name, text, label, labels);
    
public:
    MarkText();
    virtual ~MarkText();
    /**
     * @param json JSON
     */
    MarkText(const std::string& json);

public:
    /**
     * @return JSON
     */
    virtual std::string toJSON() override;

};

}

#endif // LFR_HEADER_CORE_CONFIG_MARK_HPP
