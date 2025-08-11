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
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_WUDAOZI_HPP
#define LFR_HEADER_CORE_WUDAOZI_HPP

#include "lifuren/Client.hpp"
#include "lifuren/Config.hpp"

namespace lifuren {

enum class WudaoziType {

    IMAGE, // 图片生成
    VIDEO, // 视频生成

};

struct WudaoziParams {

    int n  = 1;
    int t0 = 150;
    std::string file = "";
    std::string path = "";
    WudaoziType type = WudaoziType::VIDEO;

};

using WudaoziClient = Client<lifuren::config::ModelParams, lifuren::WudaoziParams, std::string>;

/**
 * @return 模型终端
 */
extern std::unique_ptr<lifuren::WudaoziClient> get_wudaozi_client();

}

#endif // END OF LFR_HEADER_CORE_WUDAOZI_HPP
