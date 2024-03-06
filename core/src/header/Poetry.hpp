/**
 * 诗词
 * 
 * @author acgist
 */
#pragma once

#include "Label.hpp"
#include "PoetryGC.hpp"
#include "PoetryTS.hpp"

namespace lifuren {

class Poetry;

namespace poetry {

/**
 * @param poetry 诗词
 * 
 * @return 匹配规则
 */
extern lifuren::LabelText* matchRule(const Poetry& poetry);

}

/**
 * 诗词
 */
class Poetry {

public:
    // 标题
    std::string title;
    // 作者
    std::string author;
    // 段落
    std::string segment;
    // 分词
    std:::vector<std::string> participle;
    // 规则
    LabelText* rule = nullptr;

public:
    // 段落分词
    void participleSegment();

}

}