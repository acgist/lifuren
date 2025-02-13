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

#include "torch/data.h"
#include "torch/utils.h"
#include "torch/nn/module.h"
#include "torch/serialize.h"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Logger.hpp"

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
 * @param D 数据集
 * @param L 损失函数
 * @param P 优化函数
 * @param M 模型结构
 * 
 * @author acgist
 */
template<typename D, typename L, typename P, typename M>
class Model {

protected:
    lifuren::config::ModelParams params{}; // 模型参数
    D trainDataset{ nullptr }; // 训练数据集
    D valDataset  { nullptr }; // 验证数据集
    D testDataset { nullptr }; // 测试数据集
    L loss        { nullptr }; // 损失函数
    M model       { nullptr }; // 模型结构
    std::unique_ptr<P> optimizer{ nullptr }; // 优化函数

public:
    torch::DeviceType device{ torch::DeviceType::CPU }; // 计算设备

public:
    Model(
        lifuren::config::ModelParams params = {},
        L loss  = {},
        M model = {}
    );
    virtual ~Model();

public:
    // 保存模型
    virtual bool save(const std::string& path = "./lifuren.pt");
    // 加载模型
    virtual bool load(const std::string& path = "./lifuren.pt");
    // 定义模型
    virtual bool define();
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
    virtual void train(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss);
    // 验证模型
    virtual void val(const size_t epoch);
    virtual void val(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss);
    // 测试模型
    virtual void test();
    virtual void test(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss);
    // 评估信息
    virtual void printEvaluation(const char* name, const size_t epoch, const float loss, const size_t accu_val, const size_t data_val, const size_t duration, torch::Tensor confusion_matrix);

protected:
    // 初始化权重
    virtual bool defineWeight();
    // 定义数据集
    virtual bool defineDataset() = 0;

};

} // END OF lifuren

inline void classify_evaluate(
    const torch::Tensor& target,
    const torch::Tensor& pred,
          torch::Tensor& confusion_matrix,
          size_t& accu_val,
          size_t& data_val
) {
    auto target_size = target.numel();
    auto target_pred = pred.argmax(1);
    auto accu = target_pred.eq(target).sum();
    accu_val += accu.template item<int>();
    data_val += target_size;
    int64_t* target_iter      = target.data_ptr<int64_t>();
    int64_t* target_pred_iter = target_pred.data_ptr<int64_t>();
    for (int64_t i = 0; i < target_size; ++i, ++target_iter, ++target_pred_iter) {
        confusion_matrix[*target_iter][*target_pred_iter].add_(1);
    }
}

template<typename D, typename L, typename P, typename M>
lifuren::Model<D, L, P, M>::Model(
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
    SPDLOG_DEBUG("当前计算设备：{}", torch::DeviceTypeName(this->device));
}

template<typename D, typename L, typename P, typename M>
lifuren::Model<D, L, P, M>::~Model() {
    SPDLOG_DEBUG("释放模型：{}", this->params.model_name);
}

template<typename D, typename L, typename P, typename M>
bool lifuren::Model<D, L, P, M>::save(const std::string& path) {
    if(!this->model) {
        SPDLOG_WARN("模型保存失败：没有定义模型");
        return false;
    }
    SPDLOG_DEBUG("保存模型：{}", path);
    lifuren::file::createParent(path);
    this->model->eval();
    torch::save(this->model, path);
    return true;
}

template<typename D, typename L, typename P, typename M>
bool lifuren::Model<D, L, P, M>::load(const std::string& path) {
    if(!lifuren::file::exists(path) || !lifuren::file::is_file(path)) {
        SPDLOG_WARN("加载模型失败：{}", path);
        return false;
    }
    SPDLOG_DEBUG("加载模型：{}", path);
    try {
        torch::load(this->model, path);
    } catch(const std::exception& e) {
        SPDLOG_ERROR("加载模型异常：{} - {}", path, e.what());
        return false;
    }
    this->model->to(this->device);
    this->model->eval();
    return true;
}

template<typename D, typename L, typename P, typename M>
bool lifuren::Model<D, L, P, M>::define() {
    this->model->to(this->device);
    if(!this->defineWeight()) {
        SPDLOG_WARN("初始化权重失败");
        return false;
    }
    if(!this->defineDataset()) {
        SPDLOG_WARN("定义数据集失败");
        return false;
    }
    this->print();
    return true;
}

template<typename D, typename L, typename P, typename M>
inline void lifuren::Model<D, L, P, M>::print(const bool details) {
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

template<typename D, typename L, typename P, typename M>
void lifuren::Model<D, L, P, M>::trainValAndTest(const bool val, const bool test) {
    if(!this->trainDataset) {
        SPDLOG_WARN("无效的训练数据集");
        return;
    }
    const auto a = std::chrono::system_clock::now();
    try {
        for (size_t epoch = 0; epoch < this->params.epoch_count; ++epoch) {
            this->train(epoch);
            if(val) {
                this->val(epoch);
            }
            if(this->params.check_point) {
                this->save(lifuren::file::join({
                    this->params.check_path,
                    this->params.model_name + ".checkpoint." + std::to_string(epoch) + ".pt"
                }).string());
            }
        }
        if(test) {
            this->test();
        }
    } catch(const std::exception& e) {
        SPDLOG_ERROR("训练异常：{}", e.what());
    }
    const auto z = std::chrono::system_clock::now();
    SPDLOG_DEBUG("累计耗时：{}", std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count());
}

template<typename D, typename L, typename P, typename M>
torch::Tensor lifuren::Model<D, L, P, M>::pred(const torch::Tensor& input) {
    return this->model->forward(input);
}

template<typename D, typename L, typename P, typename M>
inline void lifuren::Model<D, L, P, M>::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    pred = this->model->forward(feature);
    loss = this->loss(pred, label);
}

template<typename D, typename L, typename P, typename M>
void lifuren::Model<D, L, P, M>::train(const size_t epoch) {
    if(!this->trainDataset) {
        SPDLOG_WARN("无效的训练数据集");
        return;
    }
    size_t accu_val = 0;
    size_t data_val = 0;
    double loss_val = 0.0;
    size_t batch_count = 0;
    this->model->train();
    auto confusion_matrix = torch::zeros({ static_cast<int>(this->params.class_size), static_cast<int>(this->params.class_size) }, torch::kInt);
    const auto a = std::chrono::system_clock::now();
    for (const auto& batch : *this->trainDataset) {
        torch::Tensor pred;
        torch::Tensor loss;
        torch::Tensor data   = batch.data;
        torch::Tensor target = batch.target;
        this->train(data, target, pred, loss);
        this->optimizer->zero_grad();
        loss.backward();
        this->optimizer->step();
        if(this->params.classify) {
            classify_evaluate(target, pred, confusion_matrix, accu_val, data_val);
        }
        loss_val += loss.template item<float>();
        ++batch_count;
        if(batch_count % 10 == 0) {
            SPDLOG_INFO("当前训练第 {} 轮第 {} 批", epoch + 1, batch_count);
        }
    }
    const auto z = std::chrono::system_clock::now();
    this->printEvaluation("训练", epoch + 1, loss_val / batch_count, accu_val, data_val, std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count(), confusion_matrix);
}

template<typename D, typename L, typename P, typename M>
inline void lifuren::Model<D, L, P, M>::train(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    this->logic(feature, label, pred, loss);
}

template<typename D, typename L, typename P, typename M>
void lifuren::Model<D, L, P, M>::val(const size_t epoch) {
    if(!this->valDataset) {
        return;
    }
    size_t accu_val = 0;
    size_t data_val = 0;
    double loss_val = 0.0;
    size_t batch_count = 0;
    this->model->eval();
    auto confusion_matrix = torch::zeros({ static_cast<int>(this->params.class_size), static_cast<int>(this->params.class_size) }, torch::kInt);
    const auto a = std::chrono::system_clock::now();
    for (const auto& batch : *this->valDataset) {
        torch::Tensor pred;
        torch::Tensor loss;
        torch::Tensor data   = batch.data;
        torch::Tensor target = batch.target;
        this->val(data, target, pred, loss);
        if(this->params.classify) {
            classify_evaluate(target, pred, confusion_matrix, accu_val, data_val);
        }
        loss_val += loss.template item<float>();
        ++batch_count;
    }
    const auto z = std::chrono::system_clock::now();
    this->printEvaluation("验证", epoch + 1, loss_val / batch_count, accu_val, data_val, std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count(), confusion_matrix);
}

template<typename D, typename L, typename P, typename M>
inline void lifuren::Model<D, L, P, M>::val(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    this->logic(feature, label, pred, loss);
}

template<typename D, typename L, typename P, typename M>
void lifuren::Model<D, L, P, M>::test() {
    if(!this->testDataset) {
        return;
    }
    size_t accu_val = 0;
    size_t data_val = 0;
    double loss_val = 0.0;
    size_t batch_count = 0;
    this->model->eval();
    auto confusion_matrix = torch::zeros({ static_cast<int>(this->params.class_size), static_cast<int>(this->params.class_size) }, torch::kInt);
    const auto a = std::chrono::system_clock::now();
    for (const auto& batch : *this->testDataset) {
        torch::Tensor pred;
        torch::Tensor loss;
        torch::Tensor data   = batch.data;
        torch::Tensor target = batch.target;
        this->test(data, target, pred, loss);
        if(this->params.classify) {
            classify_evaluate(target, pred, confusion_matrix, accu_val, data_val);
        }
        loss_val += loss.template item<float>();
        ++batch_count;
    }
    const auto z = std::chrono::system_clock::now();
    this->printEvaluation("测试", 0, loss_val / batch_count, accu_val, data_val, std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count(), confusion_matrix);
}

template<typename D, typename L, typename P, typename M>
inline void lifuren::Model<D, L, P, M>::test(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    this->logic(feature, label, pred, loss);
}

template<typename D, typename L, typename P, typename M>
inline bool lifuren::Model<D, L, P, M>::defineWeight() {
    // for(auto& parameter : this->model->named_parameters()) {
    //     torch::nn::init::xavier_normal_(parameter.value());
    // }
    return true;
}

template<typename D, typename L, typename P, typename M>
inline void lifuren::Model<D, L, P, M>::printEvaluation(
    const char* name,
    const size_t epoch,
    const float loss,
    const size_t accu_val,
    const size_t data_val,
    const size_t duration,
    torch::Tensor confusion_matrix
) {
    if(this->params.classify) {
        SPDLOG_INFO(
            "当前{}第 {} 轮，损失值为：{:.6f}，正确率为：{} / {}，耗时：{}。",
            name,
            epoch,
            loss,
            accu_val,
            data_val,
            duration
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
