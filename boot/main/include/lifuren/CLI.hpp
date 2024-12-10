/**
 * CLI API
 * 
 * ./lifuren[.exe] 命令 [参数...]
 * ./lifuren[.exe] act          [act-tangxianzu  |act-guanhanqing  ] [train|pred] [model video_file|dataset model_name]
 * ./lifuren[.exe] paint        [paint-wudaozi   |paint-gukaizhi   ] [train|pred] [model image_file|dataset model_name]
 * ./lifuren[.exe] compose      [compose-shikuang|compose-liguinian] [train|pred] [model audio_file|dataset model_name]
 * ./lifuren[.exe] poetize      [poetize-lidu    |poetize-suxin    ] [train|pred] [model rhythm prompt1 prompt2|dataset model_name]
 * ./lifuren[.exe] pcm          dataset
 * ./lifuren[.exe] pepper       dataset
 * ./lifuren[.exe] embedding    [faiss|elasticsearch] dataset [pepper|ollama]
 * ./lifuren[.exe] quantization model
 * ./lifuren[.exe] [?|help]
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_BOOT_CLI_HPP
#define LFR_HEADER_BOOT_CLI_HPP

namespace lifuren {

/**
 * 命令执行
 * 
 * 执行命令行时不会启动FLTK和HTTP
 * 
 * @param argc 参数数量
 * @param argv 命令参数
 * 
 * @return 是否执行命令
 */
extern bool cli(const int argc, const char* const argv[]);

} // END OF lifuren

#endif // LFR_HEADER_BOOT_CLI_HPP
