/**
 * ONNX运行环境
 */
#ifndef LFR_HEADER_MODEL_TORCH_HPP
#define LFR_HEADER_MODEL_TORCH_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>

namespace Ort {

    struct Env;
    struct Value;
    struct Session;
    struct RunOptions;

} // END OF Ort

namespace lifuren {

/**
 * ONNX运行环境
 */
class OnnxRuntime {

public:
    const char* logid; // 日志ID
    Ort::Session   * session   { nullptr };   // ONNX会话
    Ort::RunOptions* runOptions{ nullptr };   // ONNX配置
    std::vector<const char*> inputNodeNames;  // ONNX输入参数
    std::vector<const char*> outputNodeNames; // ONNX输出参数

public:
    OnnxRuntime(const char* logid = "lifuren");
    virtual ~OnnxRuntime();

public:
    /**
     * 创建会话
     * 
     * @param path 模型路径
     * 
     * @return 是否成功
     */
    bool createSession(const std::string& path);
    bool releaseSession();
    /**
     * 运行会话
     */
    Ort::Value         runSession(float* blob, const size_t& size, const std::vector<int64_t>& inputNodeDims, std::vector<int64_t>& outputNodeDims);
    std::vector<float> runSession(float* blob, const size_t& size, const std::vector<int64_t>& inputNodeDims);

};

} // END OF lifuren

#endif // END OF LFR_HEADER_MODEL_TORCH_HPP
