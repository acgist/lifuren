/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 吴道子
 * 
 * 视频动作预测方式：
 * 上一帧图片 + 加噪 = 上一帧噪声图片 + 动作向量 = 下一帧噪声图片 -> 降噪 = 下一帧图片
 * 
 * 动作向量生成方式：
 * 1. 通过已有视频
 * 2. 通过图片生成
 * 3. 通过音频生成
 * 
 * TODO: 补帧、超分辨率
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_WUDAOZI_HPP
#define LFR_HEADER_CORE_WUDAOZI_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

using WudaoziClient = Client<lifuren::config::ModelParams, std::string, std::string>;

/**
 * @return 模型终端
 */
extern std::unique_ptr<lifuren::WudaoziClient> get_wudaozi_client();

}

#endif // END OF LFR_HEADER_CORE_WUDAOZI_HPP
