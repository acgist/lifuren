#include "lifuren/Model.hpp"

#include <array>
#include <numeric>
#include <algorithm>

#include "ggml.h"

#include "spdlog/spdlog.h"

#include "lifuren/Files.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Tensors.hpp"
#include "lifuren/Datasets.hpp"

lifuren::Model::Model(ModelParams params) : params(params) {
}

lifuren::Model::~Model() {
    ggml_free(this->ctx_weight);
    ggml_free(this->ctx_compute);
    free(this->buf_weight);
    free(this->buf_compute);
}

void lifuren::Model::initContext() {
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

bool lifuren::Model::save(const std::string& path, const std::string& filename) {
    if(this->weights.empty()) {
        SPDLOG_WARN("保存模型失败：权重为空");
        return false;
    }
    const std::string fullpath = lifuren::files::join({ path, filename }).string();
    SPDLOG_DEBUG("保存模型：{}", fullpath);
    gguf_context* gguf_ctx = gguf_init_empty();
    std::vector<const char*> wnames;
    for(const auto& pair : this->weights) {
        SPDLOG_DEBUG("保存模型权重：{}", pair.first);
        wnames.push_back(pair.second->name);
        gguf_add_tensor(gguf_ctx, pair.second);
    }
    gguf_set_arr_str(gguf_ctx, "lifuren.wnames", wnames.data(), wnames.size());
    gguf_write_to_file(gguf_ctx, fullpath.c_str(), false);
    gguf_free(gguf_ctx);
    return true;
}

lifuren::Model& lifuren::Model::load(const std::string& path, const std::string& filename) {
    this->initContext();
    const std::string fullpath = lifuren::files::join({ path, filename }).string();
    SPDLOG_DEBUG("加载模型：{}", fullpath);
    gguf_init_params params = {
        .no_alloc = false,
        .ctx      = &this->ctx_weight,
    };
    gguf_context* gguf_ctx = gguf_init_from_file(fullpath.c_str(), params);
    if (!gguf_ctx) {
        SPDLOG_WARN("加载模型失败：{}", fullpath);
        gguf_free(gguf_ctx);
        return *this;
    }
    const int wnames_index = gguf_find_key(gguf_ctx, "lifuren.wnames");
    const int wnames_size  = gguf_get_arr_n(gguf_ctx, wnames_index);
    for(int i = 0; i < wnames_size; ++i) {
        const char * name   = gguf_get_arr_str(gguf_ctx, wnames_index, i);
        ggml_tensor* weight = ggml_get_tensor(this->ctx_weight, name);
        ggml_set_param(this->ctx_compute, weight);
        this->weights[name] = weight;
        SPDLOG_DEBUG("加载模型权重：{}", name);
    }
    this->bindWeight();
    this->defineInput();
    this->defineLogits();
    this->defineLoss();
    this->defineCgraph();
    gguf_free(gguf_ctx);
    return *this;
}

bool lifuren::Model::saveEval(const std::string& path, const std::string& filename) {
    if(this->eval_gf == nullptr) {
        SPDLOG_WARN("保存模型失败：预测前向传播计算图为空");
        return false;
    }
    const std::string fullpath = lifuren::files::join({ path, filename }).string();
    SPDLOG_DEBUG("保存模型：{}", fullpath);
    ggml_graph_export(this->eval_gf, fullpath.c_str());
    return true;
}

lifuren::Model& lifuren::Model::loadEval(const std::string& path, const std::string& filename) {
    const std::string fullpath = lifuren::files::join({ path, filename }).string();
    SPDLOG_DEBUG("加载模型：{}", fullpath);
    this->eval_gf = ggml_graph_import(fullpath.c_str(), &this->ctx_weight, &this->ctx_compute);
    if(!this->eval_gf) {
        SPDLOG_WARN("加载模型失败：{}", fullpath);
        return *this;
    }
    for (int index = 0; index < this->eval_gf->n_leafs; ++index) {
        ggml_tensor* weight = this->eval_gf->leafs[index];
        const char * name   = weight->name;
        this->weights[name] = weight;
        SPDLOG_DEBUG("加载模型权重：{}", name);
    }
    this->bindWeight();
    this->datas  = ggml_graph_get_tensor(this->eval_gf, "global.datas");
    this->labels = ggml_graph_get_tensor(this->eval_gf, "global.labels");
    this->logits = ggml_graph_get_tensor(this->eval_gf, "global.logits");
    this->loss   = ggml_graph_get_tensor(this->eval_gf, "global.loss");
    return *this;
}

lifuren::Model& lifuren::Model::define(const InitType type, float mean, float sigma, float value) {
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

lifuren::Model& lifuren::Model::initWeight(InitType type, float mean, float sigma, float value) {
    if(this->weights.empty()) {
        SPDLOG_WARN("初始化权重失败：权重为空");
        return *this;
    }
    if(type == InitType::RAND) {
        for(const auto& pair : this->weights) {
            lifuren::tensors::fillRand(pair.second, mean, sigma);
        }
    } else {
        for(const auto& pair : this->weights) {
            lifuren::tensors::fill(pair.second, type == InitType::ZERO ? 0.0F : value);
        }
    }
    return *this;
}

lifuren::Model& lifuren::Model::defineInput() {
    if(this->datas == nullptr) {
        this->datas = this->buildDatas();
        ggml_set_input(this->datas);
        ggml_set_name(this->datas, "global.datas");
    } else {
        SPDLOG_WARN("重复定义输入数据");
    }
    if(this->labels == nullptr) {
        this->labels = this->buildLabels();
        ggml_set_input(this->labels);
        ggml_set_name(this->labels, "global.labels");
    } else {
        SPDLOG_WARN("重复定义输入标签");
    }
    return *this;
}

lifuren::Model& lifuren::Model::defineLogits() {
    if(this->logits == nullptr) {
        this->logits = this->buildLogits();
        ggml_set_output(this->logits);
        ggml_set_name(this->logits, "global.logits");
    } else {
        SPDLOG_WARN("重复定义计算逻辑");
    }
    return *this;
}

lifuren::Model& lifuren::Model::defineLoss() {
    if(this->loss == nullptr) {
        this->loss = this->buildLoss();
        ggml_set_output(this->loss);
        ggml_set_name(this->loss, "global.loss");
    } else {
        SPDLOG_WARN("重复定义损失函数");
    }
    return *this;
}

lifuren::Model& lifuren::Model::defineCgraph() {
    if(this->trainDataset) {
        if(this->train_gf == nullptr) {
            SPDLOG_DEBUG("定义训练前向传播计算图");
            this->train_gf = ggml_new_graph_custom(this->ctx_compute, this->params.size_cgraph, true);
            ggml_build_forward_expand(this->train_gf, this->loss);
        } else {
            SPDLOG_DEBUG("重复定义训练前向传播计算图");
        }
        if(this->train_gb == nullptr) {
            SPDLOG_DEBUG("定义训练反向传播计算图");
            this->train_gb = ggml_graph_dup(this->ctx_compute, this->train_gf);
            ggml_build_backward_expand(this->ctx_compute, this->train_gf, this->train_gb, true);
        } else {
            SPDLOG_DEBUG("重复定义训练反向传播计算图");
        }
    } else {
        SPDLOG_DEBUG("没有训练数据集忽略定义训练反向传播计算图");
    }
    if(this->valDataset) {
        if(this->val_gf == nullptr) {
            SPDLOG_DEBUG("定义验证前向传播计算图");
            this->val_gf = ggml_new_graph(this->ctx_compute);
            ggml_build_forward_expand(this->val_gf, this->loss);
        } else {
            SPDLOG_DEBUG("重复定义验证前向传播计算图");
        }
    } else {
        SPDLOG_DEBUG("没有验证数据集忽略定义验证前向传播计算图");
    }
    if(this->testDataset) {
        if(this->test_gf == nullptr) {
            SPDLOG_DEBUG("定义测试前向传播计算图");
            this->test_gf = ggml_new_graph(this->ctx_compute);
            ggml_build_forward_expand(this->test_gf, this->loss);
        } else {
            SPDLOG_DEBUG("重复定义测试前向传播计算图");
        }
    } else {
        SPDLOG_DEBUG("没有测试数据集忽略定义测试前向传播计算图");
    }
    // 必须生成推理计算图
    if(this->eval_gf == nullptr) {
        SPDLOG_DEBUG("定义推理前向传播计算图");
        this->eval_gf = ggml_new_graph(this->ctx_compute);
        ggml_build_forward_expand(this->eval_gf, this->loss);
    } else {
        SPDLOG_DEBUG("重复定义推理前向传播计算图");
    }
    return *this;
}

lifuren::Model& lifuren::Model::print() {
    std::string message{ "\n" };
    message += fmt::format(
        "{: <32s} {: <4s} {: <4s} {: <4s} {: <4s} {: <4s} {: <4s} {: <4s} {: <8s} {: <8s} {: <8s} {: <8s} {: <16s} {: <32s}\n",
        "NAME",   "FROM", "TYPE", "DIMS", "NE0",  "NE1",  "NE2",  "NE3",  "NB0",  "NB1",  "NB2",  "NB3",  "DATA",  "OP"
    );
    auto* weightTensor = ggml_get_first_tensor(this->ctx_weight);
    while(weightTensor) {
        this->print("CTXW", weightTensor, message);
        weightTensor = ggml_get_next_tensor(this->ctx_weight, weightTensor);
    }
    message += '\n';
    message += fmt::format(
        "{: <32s} {: <4s} {: <4s} {: <4s} {: <4s} {: <4s} {: <4s} {: <4s} {: <8s} {: <8s} {: <8s} {: <8s} {: <16s} {: <32s}\n",
        "NAME",   "FROM", "TYPE", "DIMS", "NE0",  "NE1",  "NE2",  "NE3",  "NB0",  "NB1",  "NB2",  "NB3",  "DATA",  "OP"
    );
    auto* computeTensor = ggml_get_first_tensor(this->ctx_compute);
    while(computeTensor) {
        this->print("CTXC", computeTensor, message);
        computeTensor = ggml_get_next_tensor(this->ctx_compute, computeTensor);
    }
    this->print("train_gf", this->train_gf, message);
    this->print("train_gb", this->train_gb, message);
    this->print("val_gf",   this->val_gf  , message);
    this->print("test_gf",  this->test_gf , message);
    this->print("eval_gf",  this->eval_gf , message);
    SPDLOG_DEBUG("\n{}\n", message);
    return *this;
}

lifuren::Model& lifuren::Model::print(const char* name, const ggml_cgraph* cgraph, std::string& message) {
    if(cgraph == nullptr) {
        return *this;
    }
    uint64_t nbytes_pad = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        nbytes_pad += ggml_nbytes_pad(cgraph->nodes[i]);
    }
    message += fmt::format(R"(
name       : {}
magic      : {}
version    : {}
leafs      : {}
nodes      : {}
nbytes_pad : {}

)",
    name,
    GGML_FILE_MAGIC,
    GGML_FILE_VERSION,
    cgraph->n_leafs,
    cgraph->n_nodes,
    nbytes_pad
    );
    message += fmt::format(
        "{: <32s} {: <4s} {: <4s} {: <4s} {: <4s} {: <4s} {: <4s} {: <4s} {: <8s} {: <8s} {: <8s} {: <8s} {: <16s} {: <32s}\n",
        "NAME",   "FROM", "TYPE", "DIMS", "NE0",  "NE1",  "NE2",  "NE3",  "NB0",  "NB1",  "NB2",  "NB3",  "DATA",  "OP"
    );
    for (int i = 0; i < cgraph->n_leafs; ++i) {
        const ggml_tensor* tensor = cgraph->leafs[i];
        this->print("IN", tensor, message);
    }
    message += '\n';
    message += fmt::format(
        "{: <32s} {: <4s} {: <4s} {: <4s} {: <4s} {: <4s} {: <4s} {: <4s} {: <8s} {: <8s} {: <8s} {: <8s} {: <16s} {: <32s}\n",
        "NAME",   "FROM", "TYPE", "DIMS", "NE0",  "NE1",  "NE2",  "NE3",  "NB0",  "NB1",  "NB2",  "NB3",  "DATA",  "OP"
    );
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        {
            const ggml_tensor* tensor = cgraph->nodes[i];
            this->print("DST", tensor, message);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor* tensor = cgraph->nodes[i]->src[j];
            if (tensor == nullptr) {
                break;
            } else {
                this->print("SRC", tensor, message);
            }
        }
        message += '\n';
    }
    return *this;
}

lifuren::Model& lifuren::Model::print(const char* from, const ggml_tensor* tensor, std::string& message) {
    if(tensor == nullptr) {
        return *this;
    }
    const int64_t* ne = tensor->ne;
    const size_t * nb = tensor->nb;
    message += fmt::format(
        "{: <32s} {: <4s} {: <4s} {: <4d} {: <4d} {: <4d} {: <4d} {: <4d} {: <8d} {: <8d} {: <8d} {: <8d} {: <16s} {: <32s}\n",
        tensor->name,
        from,
        ggml_type_name(tensor->type),
        ggml_n_dims(tensor),
        ne[0], ne[1], ne[2], ne[3],
        nb[0], nb[1], nb[2], nb[3],
        "-",
        // static_cast<const char*>(tensor->data),
        ggml_op_name(tensor->op)
    );
    return *this;
}

void lifuren::Model::train(size_t epoch, ggml_opt_context* opt_ctx) {
    if(!this->trainDataset) {
        return;
    }
    const int64_t epoch_start_us = ggml_time_us();
    const size_t count = this->trainDataset->getCount();
    const size_t batchCount = this->trainDataset->getBatchCount();
    std::vector<float> loss{};
    loss.reserve(batchCount);
    size_t accuSize = 0;
    for (size_t batch = 0; batch < batchCount; ++batch) {
        size_t size = this->trainDataset->batchGet(
            batch,
            static_cast<float*>(this->datas->data),
            static_cast<float*>(this->labels->data)
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
    const int64_t epoch_total_us = (ggml_time_us() - epoch_start_us) / 1000;
    if(this->params.classify) {
        SPDLOG_INFO("当前训练第 {} 轮，损失值为：{}，正确率为：{} / {}，耗时：{}。", epoch, meanLoss, accuSize, count, epoch_total_us);
    } else {
        SPDLOG_INFO("当前训练第 {} 轮，损失值为：{}，耗时：{}。", epoch, meanLoss, epoch_total_us);
    }
}

void lifuren::Model::val(size_t epoch) {
    if(!this->valDataset) {
        return;
    }
    const int64_t epoch_start_us = ggml_time_us();
    const size_t count = this->valDataset->getCount();
    const size_t batchCount = this->valDataset->getBatchCount();
    std::vector<float> loss{};
    loss.reserve(batchCount);
    size_t accuSize = 0;
    for (size_t batch = 0; batch < batchCount; ++batch) {
        size_t size = this->valDataset->batchGet(
            batch,
            static_cast<float*>(this->datas->data),
            static_cast<float*>(this->labels->data)
        );
        ggml_graph_compute_with_ctx(this->ctx_compute, this->val_gf, this->params.thread_size);
        loss.push_back(*ggml_get_data_f32(this->loss));
        // 类别统计正确数量
        if(this->params.classify) {
            accuSize += this->batchAccu(size);
        }
    }
    const float meanLoss = std::accumulate(loss.begin(), loss.end(), 0.0) / batchCount;
    const int64_t epoch_total_us = (ggml_time_us() - epoch_start_us) / 1000;
    if(this->params.classify) {
        SPDLOG_INFO("当前验证第 {} 轮，损失值为：{}，正确率为：{} / {}，耗时：{}。", epoch, meanLoss, accuSize, count, epoch_total_us);
    } else {
        SPDLOG_INFO("当前验证第 {} 轮，损失值为：{}，耗时：{}。", epoch, meanLoss, epoch_total_us);
    }
}

void lifuren::Model::test() {
    if(!this->testDataset) {
        return;
    }
    const int64_t epoch_start_us = ggml_time_us();
    const size_t count = this->testDataset->getCount();
    const size_t batchCount = this->testDataset->getBatchCount();
    std::vector<float> loss{};
    loss.reserve(batchCount);
    size_t accuSize = 0;
    for (size_t batch = 0; batch < batchCount; ++batch) {
        size_t size = this->testDataset->batchGet(
            batch,
            static_cast<float*>(this->datas->data),
            static_cast<float*>(this->labels->data)
        );
        ggml_graph_compute_with_ctx(this->ctx_compute, this->val_gf, this->params.thread_size);
        loss.push_back(*ggml_get_data_f32(this->loss));
        // 类别统计正确数量
        if(this->params.classify) {
            accuSize += this->batchAccu(size);
        }
    }
    const float meanLoss = std::accumulate(loss.begin(), loss.end(), 0.0) / batchCount;
    const int64_t epoch_total_us = (ggml_time_us() - epoch_start_us) / 1000;
    if(this->params.classify) {
        SPDLOG_INFO("当前测试损失值为：{}，正确率为：{} / {}，耗时：{}。", meanLoss, accuSize, count, epoch_total_us);
    } else {
        SPDLOG_INFO("当前测试损失值为：{}，耗时：{}。", meanLoss, epoch_total_us);
    }
}

float* lifuren::Model::eval(const float* input, float* output, size_t size_data) {
    if(!this->eval_gf) {
        return nullptr;
    }
    memcpy(this->datas->data, input, std::min(sizeof(float) * size_data, ggml_nbytes(this->datas)));
    ggml_graph_compute_with_ctx(this->ctx_compute, this->eval_gf, this->params.thread_size);
    const float* source = ggml_get_data_f32(this->logits);
    memcpy(output, source, sizeof(float) * size_data);
    return output;
}

std::vector<size_t> lifuren::Model::evalClassify(const float* input, size_t size_data) {
    std::vector<size_t> vector{};
    vector.reserve(this->params.size_classify);
    float *target = new float[size_data];
    float* data = this->eval(input, target, size_data);
    for (size_t index = 0; index < size_data; ++index) {
        const float* pos = data + index * this->params.size_classify;
        vector.push_back(std::distance(std::max_element(pos, pos + this->params.size_classify), pos));
    }
    delete target;
    return vector;
}

void lifuren::Model::trainAndVal() {
    if(!this->trainDataset) {
        SPDLOG_WARN("无效的训练数据集");
        return;
    }
    if(this->params.batch_size != this->trainDataset->getBatchSize()) {
        SPDLOG_WARN("批量大小错误：{} - {}", this->params.batch_size, this->trainDataset->getBatchSize());
        return;
    }
    ggml_opt_context opt_ctx{};
    this->buildOptimizer(&opt_ctx);
    const int64_t train_start_us = ggml_time_us();
    for(size_t epoch = 0; epoch < this->params.epoch_count; ++epoch) {
        this->train(epoch, &opt_ctx);
        this->val(epoch);
    }
    this->test();
    const int64_t train_total_us = ggml_time_us() - train_start_us;
    SPDLOG_DEBUG("累计耗时：{}", train_total_us);
}

void lifuren::Model::buildOptimizer(ggml_opt_context* opt_ctx) {
    ggml_opt_params opt_pars = ggml_opt_default_params(GGML_OPT_TYPE_ADAM);
    // ggml_opt_params opt_pars = ggml_opt_default_params(GGML_OPT_TYPE_LBFGS);
    opt_pars.print_forward_graph  = false;
    opt_pars.print_backward_graph = false;
    opt_pars.n_threads   = this->params.thread_size;
    opt_pars.adam.n_iter = this->params.optimizerParams.n_iter;
    // opt_pars.lbfgs.n_iter = this->params.optimizerParams.n_iter;
    ggml_opt_init(this->ctx_compute, opt_ctx, opt_pars, 0);
}

size_t lifuren::Model::batchAccu(const size_t& size) {
    size_t count = 0L;
    for (size_t index = 0; index < size; ++index) {
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
