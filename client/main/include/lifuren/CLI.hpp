/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * CLI API
 * 
 * ./lifuren[.exe] 命令 [参数...]
 * ./lifuren[.exe] audio [audio-shikuang] [pred|train] [model audio_file|dataset model_name]
 * ./lifuren[.exe] video [video-wudaozi ] [pred|train] [model video_file|dataset model_name]
 * ./lifuren[.exe] embedding dataset [audio|video]
 * ./lifuren[.exe] [?|help]
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CLIENT_CLI_HPP
#define LFR_HEADER_CLIENT_CLI_HPP

namespace lifuren {

/**
 * 执行命令
 * 
 * @return 是否执行命令成功
 */
extern bool cli(
    const int         argc,  // 参数数量
    const char* const argv[] // 命令参数
);

} // END OF lifuren

#endif // LFR_HEADER_CLIENT_CLI_HPP
