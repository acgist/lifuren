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

enum class WudaoziType {

    RESET,
    IMAGE,
    VIDEO,

};

struct WudaoziParams {

    int size = 1;
    std::string file;
    std::string path;
    WudaoziType type;

};

using WudaoziClient = Client<lifuren::config::ModelParams, WudaoziParams, std::string>;

/**
 * @return 模型终端
 */
extern std::unique_ptr<lifuren::WudaoziClient> get_wudaozi_client();

}

#endif // END OF LFR_HEADER_CORE_WUDAOZI_HPP
