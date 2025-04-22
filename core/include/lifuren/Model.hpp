/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 模型
 * 
 * 模型输出向量不要使用任何逻辑判断语句
 * 
 * https://pytorch.org/cppdocs/
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_MODEL_HPP
#define LFR_HEADER_CORE_MODEL_HPP

#include <array>
#include <memory>
#include <string>
#include <thread>
#include <concepts>

#include "torch/nn.h"
#include "torch/serialize.h"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Logger.hpp"
#include "lifuren/Dataset.hpp"

namespace lifuren {

// 损失函数
template<typename T>
concept L = std::derived_from<T, torch::nn::Module>;

// 模型结构
template<typename T>
concept M = std::derived_from<T, torch::nn::Module>;

/**
 * 模型
 * 
 * @param L 损失函数
 * @param P 优化函数
 * @param M 模型结构
 * @param D 数据集
 * 
 * @author acgist
 */
template<typename L, typename P, typename M, typename D>
class Model {

protected:
    lifuren::config::ModelParams params{}; // 模型参数
    D trainDataset{ nullptr }; // 训练数据集
    D valDataset  { nullptr }; // 验证数据集
    D testDataset { nullptr }; // 测试数据集
    L loss  { nullptr }; // 损失函数
    M model { nullptr }; // 模型实现
    std::unique_ptr<P> optimizer{ nullptr }; // 优化函数

public:
    torch::DeviceType device{ torch::DeviceType::CPU }; // 计算设备

public:
    /**
     * @param params 模型参数
     * @param loss   损失函数
     * @param model  模型实现
     */
    Model(
        lifuren::config::ModelParams params = {},
        L loss  = {},
        M model = {}
    );
    virtual ~Model();

public:
    // 保存模型
    virtual bool save(const std::string& path = "./lifuren.pt", torch::DeviceType device = torch::DeviceType::CPU);
    // 加载模型
    virtual bool load(const std::string& path = "./lifuren.pt", torch::DeviceType device = torch::DeviceType::CPU);
    // 定义模型
    virtual bool define(const bool define_weight = true, const bool define_dataset = true);
    // 打印模型
    virtual void print(const bool details = false);
    // 训练模型
    virtual void trainValAndTest(const bool val = true, const bool test = true);
    // 模型预测
    virtual torch::Tensor pred(const torch::Tensor& input);
    
protected:
    // 计算逻辑
    virtual void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss);
    // 训练模型
    virtual void train(const size_t epoch);
    // 验证模型
    virtual void val(const size_t epoch);
    // 测试模型
    virtual void test();
    /**
     * 评估信息
     * 
     * @param name  评估名称
     * @param epoch 轮次
     * @param loss  损失
     * @param accu_val 正确数量
     * @param data_val 正反总量
     * @param duration 时间消耗
     * @param confusion_matrix 混淆矩阵
     */
    virtual void printEvaluation(
        const char*  name,
        const size_t epoch,
        const float  loss,
        const size_t accu_val,
        const size_t data_val,
        const size_t duration,
        torch::Tensor confusion_matrix
    );

protected:
    // 初始化权重
    virtual void defineWeight();
    // 定义数据集
    virtual void defineDataset() = 0;

};

} // END OF lifuren

/**
 * 混淆矩阵
 * 
 * @param target 目标
 * @param pred   预测
 * @param confusion_matrix 混淆矩阵
 * @param accu_val 正确数量
 * @param data_val 正反总量
 */
inline void classify_evaluate(
    const torch::Tensor& target,
    const torch::Tensor& pred,
          torch::Tensor& confusion_matrix,
          size_t& accu_val,
          size_t& data_val
) {
    torch::NoGradGuard no_grad_guard;
    auto target_index = target.argmax(1).to(torch::kCPU);
    auto pred_index   = torch::log_softmax(pred, 1).argmax(1).to(torch::kCPU);
    // auto target_index = target.detach().argmax(1).to(torch::kCPU);
    // auto pred_index   = torch::log_softmax(pred.detach(), 1).argmax(1).to(torch::kCPU);
    auto batch_size = pred_index.numel();
    auto accu = pred_index.eq(target_index).sum();
    accu_val += accu.template item<int>();
    data_val += batch_size;
    int64_t* target_index_iter = target_index.data_ptr<int64_t>();
    int64_t* pred_index_iter   = pred_index.data_ptr<int64_t>();
    for (int64_t i = 0; i < batch_size; ++i, ++target_index_iter, ++pred_index_iter) {
        confusion_matrix[*target_index_iter][*pred_index_iter].add_(1);
    }
}

template<typename L, typename P, typename M, typename D>
lifuren::Model<L, P, M, D>::Model(
    lifuren::config::ModelParams params,
    L loss,
    M model
) : params(std::move(params)),
    loss(std::move(loss)),
    model(std::move(model)),
    device(lifuren::getDevice())
{
    if(this->model) {
        this->optimizer = std::make_unique<P>(this->model->parameters(), this->params.lr);
    } else {
        SPDLOG_WARN("没有定义模型");
    }
    if(this->params.thread_size == 0) {
        this->params.thread_size = std::thread::hardware_concurrency();
    }
    torch::set_num_threads(this->params.thread_size);
    SPDLOG_DEBUG("定义模型：{}", this->params.model_name);
    SPDLOG_DEBUG("计算设备：{}", torch::DeviceTypeName(this->device));
}

template<typename L, typename P, typename M, typename D>
lifuren::Model<L, P, M, D>::~Model() {
    SPDLOG_DEBUG("释放模型：{}", this->params.model_name);
}

template<typename L, typename P, typename M, typename D>
bool lifuren::Model<L, P, M, D>::save(const std::string& path, torch::DeviceType device) {
    if(!this->model) {
        SPDLOG_WARN("模型保存失败：没有定义模型");
        return false;
    }
    lifuren::file::createParent(path);
    this->model->eval();
    this->model->to(device);
    torch::save(this->model, path);
    SPDLOG_INFO("保存模型：{}", path);
    return true;
}

template<typename L, typename P, typename M, typename D>
bool lifuren::Model<L, P, M, D>::load(const std::string& path, torch::DeviceType device) {
    if(!lifuren::file::exists(path) || !lifuren::file::is_file(path)) {
        SPDLOG_WARN("加载模型失败：{}", path);
        return false;
    }
    SPDLOG_INFO("加载模型：{}", path);
    try {
        torch::load(this->model, path, device);
    } catch(const std::exception& e) {
        SPDLOG_ERROR("加载模型异常：{} - {}", path, e.what());
        return false;
    }
    this->model->to(this->device);
    this->model->eval();
    return true;
}

template<typename L, typename P, typename M, typename D>
bool lifuren::Model<L, P, M, D>::define(const bool define_weight, const bool define_dataset) {
    if(define_weight) {
        this->defineWeight();
    }
    if(define_dataset) {
        this->defineDataset();
    }
    this->model->to(this->device);
    this->print();
    return true;
}

template<typename L, typename P, typename M, typename D>
inline void lifuren::Model<L, P, M, D>::print(const bool details) {
    size_t total_numel = 0;
    for(const auto& parameter : this->model->named_parameters()) {
        total_numel += parameter.value().numel();
        SPDLOG_DEBUG("模型参数数量: {} = {}", parameter.key(), parameter.value().numel());
        if(details) {
            lifuren::logTensor(parameter.key(), parameter.value());
        }
    }
    SPDLOG_DEBUG("模型参数总量: {}", total_numel);
}

template<typename L, typename P, typename M, typename D>
void lifuren::Model<L, P, M, D>::trainValAndTest(const bool val, const bool test) {
    if(!this->trainDataset) {
        SPDLOG_WARN("无效的训练数据集");
        return;
    }
    SPDLOG_INFO("开始训练：{}", this->params.model_name);
    const auto a = std::chrono::system_clock::now();
    try {
        for (size_t epoch = 0; epoch < this->params.epoch_size; ++epoch) {
            this->train(epoch);
            if(val) {
                this->val(epoch);
            }
            if(this->params.check_point) {
                this->save(lifuren::file::join({
                    this->params.model_path,
                    this->params.model_name + ".checkpoint." + std::to_string(epoch) + ".ckpt"
                }).string());
            }
        }
        if(test) {
            this->test();
        }
    } catch(const std::exception& e) {
        SPDLOG_ERROR("训练异常：{}", e.what());
    } catch(...) {
        SPDLOG_ERROR("训练异常");
    }
    const auto z = std::chrono::system_clock::now();
    SPDLOG_INFO("训练完成：{}", std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count());
}

template<typename L, typename P, typename M, typename D>
torch::Tensor lifuren::Model<L, P, M, D>::pred(const torch::Tensor& input) {
    return this->model->forward(input);
}

template<typename L, typename P, typename M, typename D>
inline void lifuren::Model<L, P, M, D>::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    pred = this->model->forward(feature);
    loss = this->loss(pred, label);
}

template<typename L, typename P, typename M, typename D>
void lifuren::Model<L, P, M, D>::train(const size_t epoch) {
    if(!this->trainDataset) {
        SPDLOG_WARN("无效的训练数据集");
        return;
    }
    size_t accu_val = 0;
    size_t data_val = 0;
    double loss_val = 0.0;
    size_t batch_count = 0;
    this->model->train();
    auto confusion_matrix = torch::zeros({ static_cast<int>(this->params.class_size), static_cast<int>(this->params.class_size) }, torch::kInt).requires_grad_(false).to(this->device);
    const auto a = std::chrono::system_clock::now();
    for (const auto& batch : *this->trainDataset) {
        torch::Tensor pred;
        torch::Tensor loss;
        torch::Tensor data   = batch.data;
        torch::Tensor target = batch.target;
        this->logic(data, target, pred, loss);
        this->optimizer->zero_grad();
        loss.backward();
        this->optimizer->step();
        if(this->params.classify) {
            classify_evaluate(target, pred, confusion_matrix, accu_val, data_val);
        }
        loss_val += loss.template item<float>();
        ++batch_count;
    }
    const auto z = std::chrono::system_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count();
    this->printEvaluation("训练", epoch + 1, loss_val / batch_count, accu_val, data_val, duration, confusion_matrix);
}

template<typename L, typename P, typename M, typename D>
void lifuren::Model<L, P, M, D>::val(const size_t epoch) {
    if(!this->valDataset) {
        return;
    }
    size_t accu_val = 0;
    size_t data_val = 0;
    double loss_val = 0.0;
    size_t batch_count = 0;
    this->model->eval();
    auto confusion_matrix = torch::zeros({ static_cast<int>(this->params.class_size), static_cast<int>(this->params.class_size) }, torch::kInt).requires_grad_(false).to(this->device);
    const auto a = std::chrono::system_clock::now();
    for (const auto& batch : *this->valDataset) {
        torch::Tensor pred;
        torch::Tensor loss;
        torch::Tensor data   = batch.data;
        torch::Tensor target = batch.target;
        this->logic(data, target, pred, loss);
        if(this->params.classify) {
            classify_evaluate(target, pred, confusion_matrix, accu_val, data_val);
        }
        loss_val += loss.template item<float>();
        ++batch_count;
    }
    const auto z = std::chrono::system_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count();
    this->printEvaluation("验证", epoch + 1, loss_val / batch_count, accu_val, data_val, duration, confusion_matrix);
}

template<typename L, typename P, typename M, typename D>
void lifuren::Model<L, P, M, D>::test() {
    if(!this->testDataset) {
        return;
    }
    size_t accu_val = 0;
    size_t data_val = 0;
    double loss_val = 0.0;
    size_t batch_count = 0;
    this->model->eval();
    auto confusion_matrix = torch::zeros({ static_cast<int>(this->params.class_size), static_cast<int>(this->params.class_size) }, torch::kInt).requires_grad_(false).to(this->device);
    const auto a = std::chrono::system_clock::now();
    for (const auto& batch : *this->testDataset) {
        torch::Tensor pred;
        torch::Tensor loss;
        torch::Tensor data   = batch.data;
        torch::Tensor target = batch.target;
        this->logic(data, target, pred, loss);
        if(this->params.classify) {
            classify_evaluate(target, pred, confusion_matrix, accu_val, data_val);
        }
        loss_val += loss.template item<float>();
        ++batch_count;
    }
    const auto z = std::chrono::system_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count();
    this->printEvaluation("测试", 0, loss_val / batch_count, accu_val, data_val, duration, confusion_matrix);
}

template<typename L, typename P, typename M, typename D>
inline void lifuren::Model<L, P, M, D>::defineWeight() {
}

template<typename L, typename P, typename M, typename D>
inline void lifuren::Model<L, P, M, D>::printEvaluation(
    const char*  name,
    const size_t epoch,
    const float  loss,
    const size_t accu_val,
    const size_t data_val,
    const size_t duration,
    torch::Tensor confusion_matrix
) {
    if(this->params.classify) {
        const size_t av = this->params.first_none ? accu_val - confusion_matrix[0][0].template item<int>() : accu_val;
        const size_t dv = this->params.first_none ? data_val - confusion_matrix[0][0].template item<int>() : data_val;
        SPDLOG_INFO(
            "当前{}第 {} 轮，损失值为：{:.6f}，耗时：{}，正确率为：{} / {} = {:.6f}。",
            name,
            epoch,
            loss,
            duration,
            av,
            dv,
            1.0F * av / dv
        );
        lifuren::logTensor("混淆矩阵", confusion_matrix);
    } else {
        SPDLOG_INFO(
            "当前{}第 {} 轮，损失值为：{:.6f}，耗时：{}。",
            name,
            epoch,
            loss,
            duration
        );
    }
}

#endif // END OF LFR_HEADER_CORE_MODEL_HPP
