/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 乐谱
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_SCORE_HPP
#define LFR_HEADER_CORE_SCORE_HPP

#include "lifuren/Client.hpp"

namespace lifuren::score {

using ScoreModelClient = ModelClient<lifuren::config::ModelParams, std::string, std::string>;

/**
 * @param model 模型名称
 * 
 * @return 模型终端
 */
extern std::unique_ptr<lifuren::score::ScoreModelClient> getScoreClient(const std::string& model);

} // END OF lifuren::score

#endif // END OF LFR_HEADER_CORE_SCORE_HPP
