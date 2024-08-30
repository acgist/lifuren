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
#include <algorithm>

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

/**
 * 初始方式
 */
enum class InitType {

    ZERO,
    RAND,
    VALUE,

};

struct ModelParams {

    // 学习率
    float lr = 0.001F;
    // 批量大小
    size_t batch_size  = 100;
    // 训练次数
    size_t epoch_count = 128;
    // 线程数量
    size_t thread_size = std::thread::hardware_concurrency();
    // 分类模型
    bool classify = false;
    // 分类数量
    size_t size_classify = 0LL;
    // 计算图的大小
    size_t size_cgraph  = 16LL  * 1024;
    // 权重大小
    size_t size_weight  = 128LL * 1024 * 1024;
    // 计算大小
    size_t size_compute = 256LL * 1024 * 1024;
    // 数据类型
    ggml_type         data_type    = GGML_TYPE_F32;
    // 计算类型
    ggml_backend_type backend_type = GGML_BACKEND_TYPE_CPU;

};

protected:
    // 模型参数
    ModelParams params{};
    // 权重上下文
    void        * buf_weight { nullptr };
    ggml_context* ctx_weight { nullptr };
    // 计算上下文
    void        * buf_compute{ nullptr };
    ggml_context* ctx_compute{ nullptr };
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
    // 模型权重名称
    std::vector<std::string> names{};
    // 模型权重映射
    std::map<std::string, ggml_tensor*> weights{};

public:
    // 训练数据集
    std::unique_ptr<lifuren::datasets::Dataset> trainDataset{ nullptr };
    // 验证数据集
    std::unique_ptr<lifuren::datasets::Dataset> valDataset  { nullptr };
    // 测试数据集
    std::unique_ptr<lifuren::datasets::Dataset> testDataset { nullptr };

public:
    Model(
        ModelParams params = {}
    );
    virtual ~Model();

protected:
    // 加载上下文
    void initContext();

public:
    // 训练模型保存加载
    virtual bool   save(const std::string& path = "./", const std::string& filename = "lifuren.gguf");
    virtual Model& load(const std::string& path = "./", const std::string& filename = "lifuren.gguf");
    // 预测模型保存加载
    virtual bool   saveEval(const std::string& path = "./", const std::string& filename = "lifuren.ggml");
    virtual Model& loadEval(const std::string& path = "./", const std::string& filename = "lifuren.ggml");
    // 定义模型
    virtual Model& define(InitType type = InitType::RAND, double mean = 0.0, double sigma = 1e-2, float value = 0.0F);
    // 初始化模型
    virtual Model& defineWeight() = 0;
    // 初始化计算图
    virtual Model& defineCgraph();
    // 初始化输入
    virtual Model& defineInput();
    // 随机初始化
    virtual Model& initWeight(InitType type = InitType::RAND, double mean = 0.0, double sigma = 1e-2, float value = 0.0F);
    // 绑定模型
    virtual Model& bindWeight() = 0;
    // 打印模型
    virtual Model& print();
    virtual Model& print(ggml_cgraph* cgraph);
    virtual Model& print(const char* from, const ggml_tensor* tensor);
    // 创建损失函数
    virtual Model& defineLoss();
    virtual ggml_tensor* buildLoss()   = 0;
    // 创建计算逻辑
    virtual Model& defineLogits();
    virtual ggml_tensor* buildLogits() = 0;
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
lifuren::Model<R, I>::Model(ModelParams params) : params(params) {
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
    if(this->ctx_weight == nullptr) {
        this->buf_weight = malloc(this->params.size_weight);
        {
            ggml_init_params params = {
                .mem_size   = this->params.size_weight,
                .mem_buffer = this->buf_weight,
                .no_alloc   = false,
            };
            this->ctx_weight = ggml_init(params);
        }
    }
    if(this->ctx_compute == nullptr) {
        this->buf_compute = malloc(this->params.size_compute);
        {
            ggml_init_params params = {
                .mem_size   = this->params.size_compute,
                .mem_buffer = this->buf_compute,
                .no_alloc   = false,
            };
            this->ctx_compute = ggml_init(params);
        }
    }
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
        SPDLOG_DEBUG("保存模型权重：{}", pair.first);
        gguf_add_tensor(gguf_ctx, pair.second);
    }
    gguf_write_to_file(gguf_ctx, fullpath.string().c_str(), false);
    gguf_free(gguf_ctx);
    return true;
}

template<typename R, typename I>
lifuren::Model<R, I>& lifuren::Model<R, I>::load(const std::string& path, const std::string& filename) {
    this->initContext();
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
    SPDLOG_DEBUG("加载模型版本：{} - {}", fullpath.string(), version);
    for(const auto& name : this->names) {
        SPDLOG_DEBUG("加载模型权重：{}", name);
        ggml_tensor* weight = ggml_get_tensor(this->ctx_weight, name.c_str());
        this->weights[name] = weight;
        ggml_set_param(this->ctx_compute, weight);
    }
    this->bindWeight();
    this->defineInput();
    this->defineLogits();
    this->defineLoss();
    this->defineCgraph();
    gguf_free(gguf_ctx);
    return *this;
}

template<typename R, typename I>
bool lifuren::Model<R, I>::saveEval(const std::string& path, const std::string& filename) {
    if(this->eval_gf == nullptr) {
        SPDLOG_WARN("保存模型失败：预测前向传播计算图为空");
        return false;
    }
    std::filesystem::path fullpath = lifuren::files::join({ path, filename });
    SPDLOG_DEBUG("保存模型：{}", fullpath.string());
    ggml_graph_export(this->eval_gf, fullpath.string().c_str());
    return true;
}

template<typename R, typename I>
lifuren::Model<R, I>& lifuren::Model<R, I>::loadEval(const std::string& path, const std::string& filename) {
    std::filesystem::path fullpath = lifuren::files::join({ path, filename });
    SPDLOG_DEBUG("加载模型：{}", fullpath.string());
    this->eval_gf = ggml_graph_import(fullpath.string().c_str(), &this->ctx_weight, &this->ctx_compute);
    if(!this->eval_gf) {
        SPDLOG_WARN("加载模型失败：{}", fullpath.string());
        return *this;
    }
    for(auto& name : this->names) {
        SPDLOG_DEBUG("加载模型权重：{}", name);
        ggml_tensor* weight = ggml_graph_get_tensor(this->eval_gf, name.c_str());
        this->weights[name] = weight;
    }
    this->bindWeight();
    this->datas  = ggml_graph_get_tensor(this->eval_gf, "global.datas");
    this->labels = ggml_graph_get_tensor(this->eval_gf, "global.labels");
    this->logits = ggml_graph_get_tensor(this->eval_gf, "global.logits");
    this->loss   = ggml_graph_get_tensor(this->eval_gf, "global.loss");
    return *this;
}

template<typename R, typename I>
lifuren::Model<R, I>& lifuren::Model<R, I>::define(const InitType type, double mean, double sigma, float value) {
    this->initContext();
    this->defineWeight();
    for(const auto& pair : this->weights) {
        ggml_set_name(pair.second, pair.first.c_str());
        ggml_set_param(this->ctx_compute, pair.second);
    }
    this->initWeight(type, mean, sigma, value);
    this->defineInput();
    this->defineLogits();
    this->defineLoss();
    this->defineCgraph();
    return *this;
}

template<typename R, typename I>
lifuren::Model<R, I>& lifuren::Model<R, I>::defineCgraph() {
    // 计算图
    if(this->train_gf == nullptr) {
        SPDLOG_DEBUG("创建训练前向传播计算图");
        this->train_gf = ggml_new_graph_custom(this->ctx_compute, this->params.size_cgraph, true);
        ggml_build_forward_expand(this->train_gf, this->loss);
    }
    if(this->train_gb == nullptr) {
        SPDLOG_DEBUG("创建训练反向传播计算图");
        this->train_gb = ggml_graph_dup(this->ctx_compute, this->train_gf);
        ggml_build_backward_expand(this->ctx_compute, this->train_gf, this->train_gb, true);
    }
    if(this->valDataset && this->val_gf == nullptr) {
        SPDLOG_DEBUG("创建验证计算图");
        this->val_gf = ggml_new_graph(this->ctx_compute);
        ggml_build_forward_expand(this->val_gf, this->loss);
    }
    if(this->testDataset && this->test_gf == nullptr) {
        SPDLOG_DEBUG("创建测试前向传播计算图");
        this->test_gf = ggml_new_graph(this->ctx_compute);
        ggml_build_forward_expand(this->test_gf, this->loss);
    }
    if(this->eval_gf == nullptr) {
        SPDLOG_DEBUG("创建推理前向传播计算图");
        this->eval_gf = ggml_new_graph(this->ctx_compute);
        ggml_build_forward_expand(this->eval_gf, this->loss);
    }
    return *this;
}

template<typename R, typename I>
lifuren::Model<R, I>& lifuren::Model<R, I>::defineInput() {
    this->datas  = ggml_new_tensor_1d(this->ctx_compute, GGML_TYPE_F32, 100);
    this->labels = ggml_new_tensor_1d(this->ctx_compute, GGML_TYPE_F32, 1);
    ggml_set_input(this->datas);
    ggml_set_name(this->datas, "global.datas");
    ggml_set_input(this->labels);
    ggml_set_name(this->labels, "global.labels");
    return *this;
}

template<typename R, typename I>
lifuren::Model<R, I>& lifuren::Model<R, I>::defineLoss() {
    this->loss = this->buildLoss();
    ggml_set_output(this->loss);
    ggml_set_name(this->loss, "global.loss");
    return *this;
}

template<typename R, typename I>
lifuren::Model<R, I>& lifuren::Model<R, I>::defineLogits() {
    this->logits = this->buildLogits();
    ggml_set_output(this->logits);
    ggml_set_name(this->logits, "global.logits");
    return *this;
}

template<typename R, typename I>
lifuren::Model<R, I>& lifuren::Model<R, I>::initWeight(InitType type, double mean, double sigma, float value) {
    if(this->weights.empty()) {
        SPDLOG_WARN("随机初始失败：权重为空");
        return *this;
    }
    if(type == InitType::RAND) {
        std::random_device device{};
        std::mt19937 random{device()};
        std::normal_distribution<float> normal(mean, sigma);
        for(const auto& pair : this->weights) {
            auto tensor = pair.second;
            GGML_ASSERT(tensor->type == this->params.data_type);
            float* data{ nullptr };
            int64_t ne = 0;
            if(this->params.data_type == GGML_TYPE_F32) {
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
    } else {
        if(type == InitType::ZERO) {
            value = 0.0F;
        }
        for(const auto& pair : this->weights) {
            auto tensor = pair.second;
            GGML_ASSERT(tensor->type == this->params.data_type);
            float* data{ nullptr };
            int64_t ne = 0;
            if(this->params.data_type == GGML_TYPE_F32) {
                data = ggml_get_data_f32(tensor);
                ne   = ggml_nelements(tensor);
            } else {
                // TODO
                // SPDLOG_DEBUG("不支持的数据类型：{}", type);
            }
            for (int64_t i = 0; i < ne; ++i) {
                data[i] = value;
            }
        }
    }
    return *this;
}

template<typename R, typename I>
lifuren::Model<R, I>& lifuren::Model<R, I>::print() {
    SPDLOG_DEBUG("========== model.weight ==========");
    for(const auto& pair : this->weights) {
        SPDLOG_DEBUG("weights.{}", pair.first);
    }
    printf(
        "%-32s %-4s %-4s %-4s %-4s %-4s %-4s %-4s %-8s %-8s %-8s %-8s %-16s %-32s\n",
        "NAME", "FROM", "TYPE", "DIMS", "NE0", "NE1", "NE2", "NE3", "NB0", "NB1", "NB2", "NB3", "DATA", "OP"
    );
    auto* weightTensor = ggml_get_first_tensor(this->ctx_weight);
    while(weightTensor) {
        this->print("CTXW", weightTensor);
        weightTensor = ggml_get_next_tensor(this->ctx_weight, weightTensor);
    }
    auto* computeTensor = ggml_get_first_tensor(this->ctx_compute);
    while(computeTensor) {
        this->print("CTXC", computeTensor);
        computeTensor = ggml_get_next_tensor(this->ctx_compute, computeTensor);
    }
    SPDLOG_DEBUG("========== model.cgraph.train ==========");
    this->print(this->train_gf);
    this->print(this->train_gb);
    SPDLOG_DEBUG("========== model.cgraph.val ==========");
    this->print(this->val_gf);
    SPDLOG_DEBUG("========== model.cgraph.test ==========");
    this->print(this->test_gf);
    SPDLOG_DEBUG("========== model.cgraph.eval ==========");
    this->print(this->eval_gf);
    return *this;
}

template<typename R, typename I>
lifuren::Model<R, I>& lifuren::Model<R, I>::print(ggml_cgraph* cgraph) {
    if(cgraph == nullptr) {
        return *this;
    }
    uint64_t size_eval = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        size_eval += ggml_nbytes_pad(cgraph->nodes[i]);
    }
    {
        printf("\n");
        printf("%-16s %8x\n",   "magic",   GGML_FILE_MAGIC);
        printf("%-16s %8d\n",   "version", GGML_FILE_VERSION);
        printf("%-16s %8d\n",   "leafs",   cgraph->n_leafs);
        printf("%-16s %8d\n",   "nodes",   cgraph->n_nodes);
        printf("%-16s %8lld\n", "eval",    size_eval);
        printf("\n");
        printf(
            "%-32s %-4s %-4s %-4s %-4s %-4s %-4s %-4s %-8s %-8s %-8s %-8s %-16s %-32s\n",
            "NAME", "FROM", "TYPE", "DIMS", "NE0", "NE1", "NE2", "NE3", "NB0", "NB1", "NB2", "NB3", "DATA", "OP"
        );
        for (int i = 0; i < cgraph->n_leafs; ++i) {
            const ggml_tensor* tensor = cgraph->leafs[i];
            this->print("IN", tensor);
        }
        printf("\n");
        printf(
            "%-32s %-4s %-4s %-4s %-4s %-4s %-4s %-4s %-8s %-8s %-8s %-8s %-16s %-32s\n",
            "NAME", "FROM", "TYPE", "DIMS", "NE0", "NE1", "NE2", "NE3", "NB0", "NB1", "NB2", "NB3", "DATA", "OP"
        );
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            {
                const ggml_tensor* tensor = cgraph->nodes[i];
                this->print("DST", tensor);
            }
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                const ggml_tensor* tensor = cgraph->nodes[i]->src[j];
                if (tensor) {
                    this->print("SRC", tensor);
                } else {
                    break;
                }
            }
            printf("\n");
        }
    }
    return *this;
}

template<typename R, typename I>
lifuren::Model<R, I>& lifuren::Model<R, I>::print(const char* from, const ggml_tensor* tensor) {
    if(tensor == nullptr) {
        return *this;
    }
    const int64_t* ne = tensor->ne;
    const size_t * nb = tensor->nb;
    printf(
        "%-32s %-4s %-4s %-4d %-4lld %-4lld %-4lld %-4lld %-8zu %-8zu %-8zu %-8zu %-16p %-32s\n",
        tensor->name,
        from,
        ggml_type_name(tensor->type),
        ggml_n_dims(tensor),
        ne[0], ne[1], ne[2], ne[3],
        nb[0], nb[1], nb[2], nb[3],
        tensor->data,
        ggml_op_name(tensor->op)
    );
    return *this;
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
        if(this->params.classify) {
            accuSize += this->batchAccu(size);
        }
    }
    const float meanLoss = std::accumulate(loss.begin(), loss.end(), 0.0) / batchCount;
    const int64_t epoch_total_us = ggml_time_us() - epoch_start_us;
    if(this->params.classify) {
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
        ggml_graph_compute_with_ctx(this->ctx_compute, this->val_gf, this->params.thread_size);
        loss.push_back(*ggml_get_data_f32(this->loss));
        // 类别统计正确数量
        if(this->params.classify) {
            accuSize += this->batchAccu(size);
        }
    }
    const float meanLoss = std::accumulate(loss.begin(), loss.end(), 0.0) / batchCount;
    const int64_t epoch_total_us = ggml_time_us() - epoch_start_us;
    if(this->params.classify) {
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
        ggml_graph_compute_with_ctx(this->ctx_compute, this->val_gf, this->params.thread_size);
        loss.push_back(*ggml_get_data_f32(this->loss));
        // 类别统计正确数量
        if(this->params.classify) {
            accuSize += this->batchAccu(size);
        }
    }
    const float meanLoss = std::accumulate(loss.begin(), loss.end(), 0.0) / batchCount;
    const int64_t epoch_total_us = ggml_time_us() - epoch_start_us;
    if(this->params.classify) {
        SPDLOG_INFO("当前测试损失值为：{}，正确率为：{} / {}，耗时：{}。", meanLoss, accuSize, count, epoch_total_us);
    } else {
        SPDLOG_INFO("当前测试损失值为：{}，耗时：{}。", meanLoss, epoch_total_us);
    }
}

template<typename R, typename I>
float* lifuren::Model<R, I>::eval(const float* input, size_t size_data) {
    memcpy(this->datas->data, &input, std::min(sizeof(float) * size_data, ggml_nbytes(this->datas)));
    ggml_graph_compute_with_ctx(this->ctx_compute, this->eval_gf, this->params.thread_size);
    return ggml_get_data_f32(this->logits);
}

template<typename R, typename I>
std::vector<size_t> lifuren::Model<R, I>::evalClassify(const float* input, size_t size_data) {
    std::vector<size_t> vector{};
    vector.reserve(this->params.size_classify);
    float* data = this->eval(input, size_data);
    for (int index = 0; index < size_data; ++index) {
        const float* pos = data + index * this->params.size_classify;
        vector.push_back(std::distance(std::max_element(pos, pos + this->params.size_classify), pos));
    }
    return vector;
}

template<typename R, typename I>
void lifuren::Model<R, I>::trainAndVal() {
    ggml_opt_context opt_ctx{};
    this->buildOptimizer(&opt_ctx);
    const int64_t train_start_us = ggml_time_us();
    for(int epoch = 0; epoch < this->params.epoch_count; ++epoch) {
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
    opt_pars.n_threads   = this->params.thread_size;
    opt_pars.adam.n_iter = 1;
    ggml_opt_init(this->ctx_compute, opt_ctx, opt_pars, 0);
}

template<typename R, typename I>
size_t lifuren::Model<R, I>::batchAccu(const size_t& size) {
    size_t count = 0L;
    for (int index = 0; index < size; ++index) {
        const float* predPos = ggml_get_data_f32(this->logits) + index * this->params.size_classify;
        const size_t predClassify = std::distance(std::max_element(predPos, predPos + this->params.classify), predPos);
        const float* labelPos = ggml_get_data_f32(this->labels) + index * this->params.size_classify;
        const size_t labelClassify = std::distance(std::max_element(labelPos, labelPos + this->params.classify), labelPos);
        if(predClassify == labelClassify) {
            ++count;
        }
    }
    return count;
}

#endif // LFR_HEADER_MODEL_MODEL_HPP
