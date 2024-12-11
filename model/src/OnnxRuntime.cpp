#include "lifuren/Onnx.hpp"

#include <atomic>
#include <thread>

#include "spdlog/spdlog.h"

#include "onnxruntime_cxx_api.h"

static Ort::Env*       env      { nullptr }; // ONNX运行环境
static std::atomic_int env_count{ 0       }; // ONNX环境计数

static std::mutex mutex;

// 默认日志
static OrtLoggingLevel log_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;

lifuren::OnnxRuntime::OnnxRuntime(
    const char* logid
) : logid(logid)
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        ++env_count;
        if(!env) {
            env = new Ort::Env(log_level, logid);
        }
    }
}

lifuren::OnnxRuntime::~OnnxRuntime() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        if(env && --env_count == 0) {
            SPDLOG_DEBUG("释放ONNX运行环境：{}", this->logid);
            env->release();
            delete env;
            env = nullptr;
        }
    }
    if(this->session) {
        SPDLOG_DEBUG("释放ONNX会话：{}", this->logid);
        this->session->release();
        delete this->session;
        this->session = nullptr;
    }
    if(this->runOptions) {
        SPDLOG_DEBUG("释放ONNX配置：{}", this->logid);
        this->runOptions->release();
        delete this->runOptions;
        this->runOptions = nullptr;
    }
    for(auto ptr : this->inputNodeNames) {
        delete[] ptr;
    }
    for(auto ptr : this->outputNodeNames) {
        delete[] ptr;
    }
}

bool lifuren::OnnxRuntime::createSession(const std::string& path) {
    SPDLOG_DEBUG("创建会话：{} - {}", this->logid, path);
    Ort::SessionOptions options;
    options.SetLogSeverityLevel(static_cast<int>(log_level));
    options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    #if _WIN32
    std::wstring wPath(path.begin(), path.end());
    this->session = new Ort::Session(*env, wPath.c_str(), options);
    #else
    this->session = new Ort::Session(*env, path.c_str(), options);
    #endif
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t inputNodeCount  = this->session->GetInputCount();
    const size_t outputNodeCount = this->session->GetOutputCount();
    for(size_t index = 0; index < inputNodeCount; ++index) {
        Ort::AllocatedStringPtr name = this->session->GetInputNameAllocated(index, allocator);
        char* copy = new char[32];
        std::strcpy(copy, name.get());
        this->inputNodeNames.push_back(copy);
        SPDLOG_DEBUG("输入节点：{} - {}", index, copy);
    }
    for(size_t index = 0; index < outputNodeCount; ++ index) {
        Ort::AllocatedStringPtr name = this->session->GetOutputNameAllocated(index, allocator);
        char* copy = new char[32];
        std::strcpy(copy, name.get());
        this->outputNodeNames.push_back(copy);
        SPDLOG_DEBUG("输出节点：{} - {}", index, copy);
    }
    this->runOptions = new Ort::RunOptions(nullptr);
    return true;
}

Ort::Value lifuren::OnnxRuntime::runSession(
          float * blob,
    const size_t& size,
    const std::vector<int64_t>& inputNodeDims,
          std::vector<int64_t>& outputNodeDims
) {
    #ifdef __CUDA__
    // TODO: CUDA
    #else
    const Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        blob,
        size,
        inputNodeDims.data(),
        inputNodeDims.size()
    );
    #endif
    auto outputTensor = this->session->Run(
        *this->runOptions,
        inputNodeNames.data(),
        &inputTensor,
        inputNodeNames.size(),
        outputNodeNames.data(),
        outputNodeNames.size()
    );
    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    outputNodeDims = typeInfo.GetTensorTypeAndShapeInfo().GetShape();
    return std::move(outputTensor.front());
}

std::vector<float> lifuren::OnnxRuntime::runSession(
          float * blob,
    const size_t& size,
    const std::vector<int64_t>& inputNodeDims
) {
    std::vector<int64_t> outputNodeDims;
    auto output = this->runSession(blob, size, inputNodeDims, outputNodeDims);
    int dims = 1;
    for(const auto& dim : outputNodeDims) {
        dims *= dim;
    }
    std::vector<float> ret;
    ret.resize(dims);
    float* data = output.GetTensorMutableData<float>();
    std::memcpy(ret.data(), data, ret.size() * sizeof(float));
    return ret;
}
