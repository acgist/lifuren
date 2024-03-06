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
namespace poetry {

/**
 * @param poetry 诗词
 * 
 * @return 匹配规则
 */
extern lifuren::LabelText* matchRule(const std::string& poetry);

}
}