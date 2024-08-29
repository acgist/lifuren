/**
 * 模型
 * 
 * @author acgist
 * 
 * https://github.com/ggerganov/ggml/blob/master/examples/mnist/mnist-common.h
 * https://github.com/ggerganov/ggml/blob/master/examples/mnist/mnist-common.cpp
 * https://github.com/ggerganov/ggml/blob/master/examples/mnist/mnist-train.cpp
 * 
 * TODO:
 * 1. datas->features
 */
#ifndef LFR_HEADER_MODEL_MODEL_HPP
#define LFR_HEADER_MODEL_MODEL_HPP

#include "lifuren/Files.hpp"
#include "lifuren/Datasets.hpp"
#include "lifuren/config/Config.hpp"

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <numeric>

#include "ggml.h"

#include <random>
#include <filesystem>

#include "spdlog/spdlog.h"

namespace lifuren {

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
    // 图的大小
    size_t size_cgraph  = 16  * 1024;
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
    ggml_context * ctx_data { nullptr };
    ggml_context * ctx_eval { nullptr };
    // 模型名称
    std::vector<std::string> names{};
    // 模型权重
    std::map<std::string, ggml_tensor*> weights{};
    bool classify = false;
    size_t size_classify = 0L;
    // 损失函数
    ggml_tensor* loss  { nullptr };
    // 目标函数
    ggml_tensor* logits{ nullptr };
    // 输入数据
    ggml_tensor* datas { nullptr };
    // 输入标签
    ggml_tensor* labels{ nullptr };
    // 预测结果
    ggml_tensor* preds { nullptr };
    // 计算图
    ggml_cgraph* train_gf { nullptr };
    ggml_cgraph* train_gb { nullptr };
    ggml_cgraph* val_gf   { nullptr };
    ggml_cgraph* test_gf  { nullptr };
    ggml_cgraph* eval_gf  { nullptr };

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
        size_t size_cgraph  = 16  * 1024,
        float lr        = 0.001F,
        int batch_size  = 100,
        int epoch_count = 128,
        int thread_size = std::thread::hardware_concurrency(),
        bool classify   = false,
        size_t size_classify = 0,
        ggml_type type  = GGML_TYPE_F32,
        ggml_backend_type backend = GGML_BACKEND_TYPE_CPU
    );
    virtual ~Model();

protected:
    void initContext();

public:
    // 加载模型
    virtual Model& load(const std::string& path = "./", const std::string& filename = "lifuren.gguf");
    virtual Model& loadWeight(const std::string& path = "./", const std::string& filename = "lifuren.ggml");
    // 绑定模型
    virtual Model& bind() = 0;
    // 初始化模型
    virtual Model& init() = 0;
    // 随机初始化
    virtual Model& rand(double mean = 0.0, double sigma = 1e-2);
    // 创建计算图
    virtual Model& build();
    // 打印模型
    virtual Model& print();
    // 创建损失函数
    virtual ggml_tensor* buildLoss()   = 0;
    // 创建计算逻辑
    virtual ggml_tensor* buildLogits() = 0;
    // 保存模型
    virtual bool save(const std::string& path = "./", const std::string& filename = "lifuren.gguf");
    virtual bool saveWeight(const std::string& path = "./", const std::string& filename = "lifuren.ggml");
    // 训练模型
    virtual void train(size_t epoch, ggml_opt_context* opt_ctx);
    // 验证模型
    virtual void val(size_t epoch);
    // 测试模型
    virtual void test();
    // 模型预测
    virtual float* eval(const float* input, size_t size_data);
    // 模型预测
    virtual std::vector<size_t> evalClassify(const float* input, size_t size_data);
    // 训练验证
    virtual void trainAndVal();
    // 优化函数
    virtual void buildOptimizer(ggml_opt_context* opt_ctx);
    // 正确数量
    virtual size_t batchAccu(const size_t& size);

};

// TODO: concept

}

template<typename R, typename I>
lifuren::Model<R, I>::Model(
    size_t size_weight,
    size_t size_compute,
    size_t size_cgraph,
    float lr,
    int batch_size,
    int epoch_count,
    int thread_size,
    bool classify,
    size_t size_classify,
    ggml_type type,
    ggml_backend_type backend
) :
size_weight(size_weight),
size_compute(size_compute),
size_cgraph(size_cgraph),
lr(lr),
batch_size(batch_size),
epoch_count(epoch_count),
thread_size(thread_size),
classify(classify),
size_classify(size_classify),
type(type),
backend(backend)
{
    this->initContext();
}

template<typename R, typename I>
lifuren::Model<R, I>::~Model() {
    ggml_free(ctx_data);
    ggml_free(ctx_eval);
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
    std::filesystem::path fullpath = lifuren::files::join({ path, filename });
    SPDLOG_DEBUG("加载模型：{}", fullpath.string());
    gguf_init_params params = {
        .no_alloc = false,
        .ctx      = &this->ctx_weight,
    };
    gguf_context* gguf_ctx = gguf_init_from_file(fullpath.string().c_str(), params);
    if (!gguf_ctx) {
        SPDLOG_WARN("加载模型失败：{}", fullpath.string());
        gguf_free(gguf_ctx);
        return *this;
    }
    const char* version = gguf_get_val_str(gguf_ctx, gguf_find_key(gguf_ctx, "lifuren.version"));
    SPDLOG_DEBUG("模型版本：{} - {}", fullpath.string(), version);
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
lifuren::Model<R, I>& lifuren::Model<R, I>::loadWeight(const std::string& path, const std::string& filename) {
    std::filesystem::path fullpath = lifuren::files::join({ path, filename });
    SPDLOG_DEBUG("加载模型：{}", fullpath.string());
    this->eval_gf = ggml_graph_import(fullpath.string().c_str(), &this->ctx_data, &this->ctx_eval);
    if(!this->eval_gf) {
        SPDLOG_WARN("加载模型失败：{}", fullpath.string());
        return *this;
    }
    this->datas  = ggml_graph_get_tensor(this->eval_gf, "global.datas");
    this->labels = ggml_graph_get_tensor(this->eval_gf, "global.labels");
    this->logits = ggml_graph_get_tensor(this->eval_gf, "global.logits");
    this->loss   = ggml_graph_get_tensor(this->eval_gf, "global.loss");
    return *this;
}

template<typename R, typename I>
lifuren::Model<R, I>& lifuren::Model<R, I>::rand(double mean, double sigma) {
    if(this->weights.empty()) {
        SPDLOG_WARN("随机初始失败：权重为空");
        return *this;
    }
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
lifuren::Model<R, I>& lifuren::Model<R, I>::build() {
    if(this->weights.empty()) {
        SPDLOG_WARN("创建模型失败：权重为空");
        return *this;
    }
    // 设置名称
    for(const auto& pair : this->weights) {
        ggml_set_name(pair.second, pair.first.c_str());
        ggml_set_param(this->ctx_compute, pair.second);
    }
    SPDLOG_DEBUG("创建输入数据");
    ggml_set_input(this->datas);
    ggml_set_name(this->datas, "global.datas");
    ggml_set_input(this->labels);
    ggml_set_name(this->labels, "global.labels");
    SPDLOG_DEBUG("创建逻辑函数");
    this->logits = this->buildLogits();
    ggml_set_output(this->logits);
    ggml_set_name(this->logits, "global.logits");
    SPDLOG_DEBUG("创建损失函数");
    this->loss = this->buildLoss();
    ggml_set_output(this->loss);
    ggml_set_name(this->loss, "global.loss");
    SPDLOG_DEBUG("创建训练计算图的算子");
    this->train_gf = ggml_new_graph_custom(this->ctx_compute, this->size_cgraph, true);
    ggml_build_forward_expand(this->train_gf, this->loss);
    this->train_gb = ggml_graph_dup(this->ctx_compute, this->train_gf);
    ggml_build_backward_expand(this->ctx_compute, this->train_gf, this->train_gb, true);
    if(this->valDataset) {
        SPDLOG_DEBUG("创建验证计算图的算子");
        this->val_gf = ggml_new_graph(this->ctx_compute);
        ggml_build_forward_expand(this->val_gf, this->loss);
    }
    if(this->testDataset) {
        SPDLOG_DEBUG("创建测试计算图的算子");
        this->test_gf = ggml_new_graph(this->ctx_compute);
        ggml_build_forward_expand(this->test_gf, this->loss);
    }
    SPDLOG_DEBUG("创建预测计算图的算子");
    this->eval_gf = ggml_new_graph(this->ctx_compute);
    ggml_build_forward_expand(this->eval_gf, this->loss);
    return *this;
}

template<typename R, typename I>
lifuren::Model<R, I>& lifuren::Model<R, I>::print() {
    SPDLOG_DEBUG("========== model.weights ==========");
    for(const auto& pair : this->weights) {
        SPDLOG_DEBUG("weights.{}", pair.first);
    }
    // this->ctx_weight->objects_begin;
    return *this;
}

template<typename R, typename I>
bool lifuren::Model<R, I>::save(const std::string& path, const std::string& filename) {
    if(this->weights.empty()) {
        SPDLOG_WARN("保存模型失败：权重为空");
        return false;
    }
    std::filesystem::path fullpath = lifuren::files::join({ path, filename });
    SPDLOG_DEBUG("保存模型：{}", fullpath.string());
    gguf_context* gguf_ctx = gguf_init_empty();
    gguf_set_val_str(gguf_ctx, "lifuren.version", "1.0.0");
    for(const auto& pair : this->weights) {
        const int64_t* ne = pair.second->ne;
        SPDLOG_DEBUG("保存模型权重：{} - {} - {} - {} - {}", pair.first, ne[0], ne[1], ne[2], ne[3]);
        gguf_add_tensor(gguf_ctx, pair.second);
    }
    gguf_write_to_file(gguf_ctx, fullpath.string().c_str(), false);
    return true;
}

template<typename R, typename I>
bool lifuren::Model<R, I>::saveWeight(const std::string& path, const std::string& filename) {
    if(this->eval_gf == nullptr) {
        SPDLOG_WARN("保存权重失败：权重为空");
        return false;
    }
    std::filesystem::path fullpath = lifuren::files::join({ path, filename });
    SPDLOG_DEBUG("保存权重：{}", fullpath.string());
    ggml_graph_export(this->eval_gf, fullpath.string().c_str());
    return true;
}

template<typename R, typename I>
void lifuren::Model<R, I>::train(size_t epoch, ggml_opt_context* opt_ctx) {
    if(!this->trainDataset) {
        return;
    }
    const int64_t epoch_start_us = ggml_time_us();
    const size_t count = this->trainDataset->getCount();
    const size_t batchCount = this->trainDataset->getBatchCount();
    std::vector<float> loss{};
    loss.reserve(batchCount);
    size_t accuSize = 0;
    for (int batch = 0; batch < batchCount; ++batch) {
        size_t size = this->trainDataset->batchGet(
            batch,
            this->datas->data,  ggml_nbytes(this->datas),
            this->labels->data, ggml_nbytes(this->labels)
        );
        enum ggml_opt_result opt_result = ggml_opt_resume_g(this->ctx_compute, opt_ctx, this->loss, this->train_gf, this->train_gb, NULL, NULL);
        GGML_ASSERT(opt_result == GGML_OPT_RESULT_OK || opt_result == GGML_OPT_RESULT_DID_NOT_CONVERGE);
        loss.push_back(*ggml_get_data_f32(this->loss));
        // 类别统计正确数量
        if(this->classify) {
            accuSize += this->batchAccu(size);
        }
    }
    const float meanLoss = std::accumulate(loss.begin(), loss.end(), 0.0) / batchCount;
    const int64_t epoch_total_us = ggml_time_us() - epoch_start_us;
    if(this->classify) {
        SPDLOG_INFO("当前训练第 {} 轮，损失值为：{}，正确率为：{} / {}，耗时：{}。", epoch, meanLoss, accuSize, count, epoch_total_us);
    } else {
        SPDLOG_INFO("当前训练第 {} 轮，损失值为：{}，耗时：{}。", epoch, meanLoss, epoch_total_us);
    }
}

template<typename R, typename I>
void lifuren::Model<R, I>::val(size_t epoch) {
    if(!this->valDataset) {
        return;
    }
    const int64_t epoch_start_us = ggml_time_us();
    const size_t count = this->valDataset->getCount();
    const size_t batchCount = this->valDataset->getBatchCount();
    std::vector<float> loss{};
    loss.reserve(batchCount);
    size_t accuSize = 0;
    for (int batch = 0; batch < batchCount; ++batch) {
        size_t size = this->valDataset->batchGet(
            batch,
            this->datas->data,  ggml_nbytes(this->datas),
            this->labels->data, ggml_nbytes(this->labels)
        );
        ggml_graph_compute_with_ctx(this->ctx_compute, this->val_gf, this->thread_size);
        loss.push_back(*ggml_get_data_f32(this->loss));
        // 类别统计正确数量
        if(this->classify) {
            accuSize += this->batchAccu(size);
        }
    }
    const float meanLoss = std::accumulate(loss.begin(), loss.end(), 0.0) / batchCount;
    const int64_t epoch_total_us = ggml_time_us() - epoch_start_us;
    if(this->classify) {
        SPDLOG_INFO("当前验证第 {} 轮，损失值为：{}，正确率为：{} / {}，耗时：{}。", epoch, meanLoss, accuSize, count, epoch_total_us);
    } else {
        SPDLOG_INFO("当前验证第 {} 轮，损失值为：{}，耗时：{}。", epoch, meanLoss, epoch_total_us);
    }
}

template<typename R, typename I>
void lifuren::Model<R, I>::test() {
    if(!this->testDataset) {
        return;
    }
    const int64_t epoch_start_us = ggml_time_us();
    const size_t count = this->testDataset->getCount();
    const size_t batchCount = this->testDataset->getBatchCount();
    std::vector<float> loss{};
    loss.reserve(batchCount);
    size_t accuSize = 0;
    for (int batch = 0; batch < batchCount; ++batch) {
        size_t size = this->testDataset->batchGet(
            batch,
            this->datas->data,  ggml_nbytes(this->datas),
            this->labels->data, ggml_nbytes(this->labels)
        );
        ggml_graph_compute_with_ctx(this->ctx_compute, this->val_gf, this->thread_size);
        loss.push_back(*ggml_get_data_f32(this->loss));
        // 类别统计正确数量
        if(this->classify) {
            accuSize += this->batchAccu(size);
        }
    }
    const float meanLoss = std::accumulate(loss.begin(), loss.end(), 0.0) / batchCount;
    const int64_t epoch_total_us = ggml_time_us() - epoch_start_us;
    if(this->classify) {
        SPDLOG_INFO("当前测试损失值为：{}，正确率为：{} / {}，耗时：{}。", meanLoss, accuSize, count, epoch_total_us);
    } else {
        SPDLOG_INFO("当前测试损失值为：{}，耗时：{}。", meanLoss, epoch_total_us);
    }
}

template<typename R, typename I>
float* lifuren::Model<R, I>::eval(const float* input, size_t size_data) {
    memcpy(this->datas->data, &input, std::min(sizeof(float) * size_data, ggml_nbytes(this->datas)));
    ggml_graph_compute_with_ctx(this->ctx_compute, this->eval_gf, this->thread_size);
    return ggml_get_data_f32(this->logits);
}

template<typename R, typename I>
std::vector<size_t> lifuren::Model<R, I>::evalClassify(const float* input, size_t size_data) {
    std::vector<size_t> vector{};
    vector.reserve(this->size_classify);
    float* data = this->eval(input, size_data);
    for (int index = 0; index < size_data; ++index) {
        const float* pos = data + index * this->size_classify;
        vector.push_back(std::distance(std::max_element(pos, pos + this->size_classify), pos));
    }
    return vector;
}

template<typename R, typename I>
void lifuren::Model<R, I>::trainAndVal() {
    ggml_opt_context opt_ctx{};
    this->buildOptimizer(&opt_ctx);
    const int64_t train_start_us = ggml_time_us();
    for(int epoch = 0; epoch < this->epoch_count; ++epoch) {
        this->train(epoch, &opt_ctx);
        this->val(epoch);
    }
    this->test();
    const int64_t train_total_us = ggml_time_us() - train_start_us;
}

template<typename R, typename I>
void lifuren::Model<R, I>::buildOptimizer(ggml_opt_context* opt_ctx) {
    ggml_opt_params opt_pars = ggml_opt_default_params(GGML_OPT_TYPE_ADAM);
    opt_pars.print_forward_graph  = false;
    opt_pars.print_backward_graph = false;
    opt_pars.n_threads   = this->thread_size;
    opt_pars.adam.n_iter = 1;
    ggml_opt_init(this->ctx_compute, opt_ctx, opt_pars, 0);
}

template<typename R, typename I>
size_t lifuren::Model<R, I>::batchAccu(const size_t& size) {
    size_t count = 0L;
    for (int index = 0; index < size; ++index) {
        const float* predPos = ggml_get_data_f32(this->logits) + index * this->size_classify;
        const size_t predClassify = std::distance(std::max_element(predPos, predPos + this->classify), predPos);
        const float* labelPos = ggml_get_data_f32(this->labels) + index * this->size_classify;
        const size_t labelClassify = std::distance(std::max_element(labelPos, labelPos + this->classify), labelPos);
        if(predClassify == labelClassify) {
            ++count;
        }
    }
    return count;
}

#endif // LFR_HEADER_MODEL_MODEL_HPP
