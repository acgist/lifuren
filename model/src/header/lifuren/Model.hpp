/**
 * 模型
 * 
 * @author acgist
 * 
 * https://github.com/ggerganov/ggml/blob/master/examples/mnist/mnist-common.h
 * https://github.com/ggerganov/ggml/blob/master/examples/mnist/mnist-common.cpp
 * https://github.com/ggerganov/ggml/blob/master/examples/mnist/mnist-train.cpp
 */
#ifndef LFR_HEADER_MODEL_MODEL_HPP
#define LFR_HEADER_MODEL_MODEL_HPP

#include "lifuren/Datasets.hpp"
#include "lifuren/config/Config.hpp"

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <thread>

#include "ggml.h"

#include <random>
#include <filesystem>

#include "spdlog/spdlog.h"

namespace lifuren {

template<typename T>
struct eval_result {
    // 数据
    ggml_tensor* datas{ nullptr };
    // 标签
    std::vector<T> labels{};
    // 预测
    std::vector<T> preds {};
    // 损失
    std::vector<double> loss{};
};

/**
 * 李夫人模型
 * 
 * @author acgist
 */
template<typename R, typename I>
class Model {

protected:
    // 权重大小
    size_t size_weight  = 128 * 1024 * 1024;
    // 计算大小
    size_t size_compute = 2   * 1024 * 1024;
    // 学习率
    float lr = 0.001F;
    // 批量大小
    int batch_size  = 100;
    // 训练次数
    int epoch_count = 128;
    // 线程数量
    int thread_size = 4;
    // 数据类型
    ggml_type type = GGML_TYPE_F32;
    // 计算类型
    ggml_backend_type backend = GGML_BACKEND_TYPE_CPU;
    // 权重
    void        * buf_weight { nullptr };
    ggml_context* ctx_weight { nullptr };
    // 计算
    void        * buf_compute{ nullptr };
    ggml_context* ctx_compute{ nullptr };
    // 目标函数
    ggml_tensor* logits{ nullptr };
    // 模型名称
    std::vector<std::string> names{};
    // 模型权重
    std::map<std::string, ggml_tensor*> weights{};

public:
    // 训练数据集
    std::unique_ptr<lifuren::datasets::Dataset> trainDataset{ nullptr };
    // 验证数据集
    std::unique_ptr<lifuren::datasets::Dataset> valDataset{ nullptr };
    // 测试数据集
    std::unique_ptr<lifuren::datasets::Dataset> testDataset{ nullptr };

public:
    Model(
        size_t size_weight  = 128 * 1024 * 1024,
        size_t size_compute = 2   * 1024 * 1024,
        float lr        = 0.001F,
        int batch_size  = 100,
        int epoch_count = 128,
        int thread_size = std::thread::hardware_concurrency(),
        ggml_type type  = GGML_TYPE_F32,
        ggml_backend_type backend = GGML_BACKEND_TYPE_CPU
    );
    virtual ~Model();

protected:
    void initContext();

public:
    // 加载模型
    virtual Model& load(const std::string& path = "./", const std::string& filename = "lifuren.model");
    // 绑定模型
    virtual Model& bind() = 0;
    // 初始化模型
    virtual Model& init() = 0;
    // 随机初始化
    virtual Model& rand(double mean = 0.0, double sigma = 1e-2);
    // 创建计算图
    virtual Model& build() = 0;
    // 保存模型
    virtual bool save(const std::string& path = "./", const std::string& filename = "lifuren.model");
    // 训练模型
    virtual void train() = 0;
    // 验证模型
    virtual void val() = 0;
    // 测试模型
    virtual void test() = 0;
    // 损失函数
    virtual double loss(const eval_result<R>& result) = 0;
    // 正确数量
    virtual double accuracy(const eval_result<R>& result) = 0;
    // 模型微调
    virtual void finetune() = 0;
    // 模型量化
    virtual void quantization(ggml_type type) = 0;
    // 模型预测
    virtual R eval(I input) = 0;
    // 训练验证
    virtual void trainAndVal();

};

// TODO: concept

}

template<typename R, typename I>
lifuren::Model<R, I>::Model(
    size_t size_weight,
    size_t size_compute,
    float lr,
    int batch_size,
    int epoch_count,
    int thread_size,
    ggml_type type,
    ggml_backend_type backend
) :
size_weight(size_weight),
size_compute(size_compute),
lr(lr),
batch_size(batch_size),
epoch_count(epoch_count),
thread_size(thread_size),
type(type),
backend(backend)
{
    this->initContext();
}

template<typename R, typename I>
lifuren::Model<R, I>::~Model() {
    ggml_free(this->ctx_weight);
    ggml_free(this->ctx_compute);
    free(this->buf_weight);
    free(this->buf_compute);
}

template<typename R, typename I>
void lifuren::Model<R, I>::initContext() {
    this->buf_weight = malloc(this->size_weight);
    {
        ggml_init_params params = {
            .mem_size   = this->size_weight,
            .mem_buffer = this->buf_weight,
            .no_alloc   = false,
        };
        this->ctx_weight = ggml_init(params);
    }
    this->buf_compute = malloc(this->size_compute);
    {
        ggml_init_params params = {
            .mem_size   = this->size_compute,
            .mem_buffer = this->buf_compute,
            .no_alloc   = false,
        };
        this->ctx_compute = ggml_init(params);
    }
}

template<typename R, typename I>
lifuren::Model<R, I>& lifuren::Model<R, I>::load(const std::string& path, const std::string& filename) {
    std::filesystem::path fullpath{ path };
    if(!filename.empty()) {
        fullpath /= filename;
    }
    SPDLOG_DEBUG("加载模型：{}", filename);
    gguf_init_params params = {
        .no_alloc = false,
        .ctx      = &this->ctx_weight,
    };
    gguf_context* gguf_ctx = gguf_init_from_file(fullpath.string().c_str(), params);
    if (!gguf_ctx) {
        SPDLOG_WARN("加载模型失败：{}", filename);
        gguf_free(gguf_ctx);
        return *this;
    }
    const char* version = gguf_get_val_str(gguf_ctx, gguf_find_key(gguf_ctx, "lifuren.version"));
    SPDLOG_DEBUG("模型版本：{} - {}", filename, version);
    for(auto& name : this->names) {
        ggml_tensor* weight = ggml_get_tensor(this->ctx_weight, name.c_str());
        this->weights[name] = weight;
        const int64_t* ne = weight->ne;
        SPDLOG_DEBUG("加载模型权重：{} - {} - {} - {} - {}", name, ne[0], ne[1], ne[2], ne[3]);
    }
    // TODO: 验证释放
    gguf_free(gguf_ctx);
    return *this;
}

template<typename R, typename I>
lifuren::Model<R, I>& lifuren::Model<R, I>::rand(double mean, double sigma) {
    std::random_device device{};
    std::mt19937 random{device()};
    std::normal_distribution<float> normal(mean, sigma);
    for(const auto& pair : this->weights) {
        auto tensor = pair.second;
        GGML_ASSERT(tensor->type == this->type);
        float* data{ nullptr };
        int64_t ne = 0;
        if(this->type == GGML_TYPE_F32) {
            data = ggml_get_data_f32(tensor);
            ne   = ggml_nelements(tensor);
        } else {
            // TODO
            // SPDLOG_DEBUG("不支持的数据类型：{}", type);
        }
        for (int64_t i = 0; i < ne; ++i) {
            data[i] = normal(random);
        }
    }
    return *this;
}

template<typename R, typename I>
bool lifuren::Model<R, I>::save(const std::string& path, const std::string& filename) {
    std::filesystem::path fullpath{ path };
    if(!filename.empty()) {
        fullpath /= filename;
    }
    SPDLOG_DEBUG("保存模型：{}", filename);
    gguf_context* gguf_ctx = gguf_init_empty();
    gguf_set_val_str(gguf_ctx, "lifuren.version", "1.0.0");
    for(const auto& pair : this->weights) {
        gguf_add_tensor(gguf_ctx, pair.second);
    }
    gguf_write_to_file(gguf_ctx, fullpath.string().c_str(), false);
    return true;
}

template<typename R, typename I>
void lifuren::Model<R, I>::trainAndVal() {

}

#endif // LFR_HEADER_MODEL_MODEL_HPP
