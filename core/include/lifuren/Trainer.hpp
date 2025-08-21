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
 * https://pytorch.org/cppdocs/
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_TRAINER_HPP
#define LFR_HEADER_CORE_TRAINER_HPP

#include <memory>
#include <string>
#include <thread>
#include <concepts>

#include "torch/nn.h"
#include "torch/optim.h"
#include "torch/serialize.h"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Dataset.hpp"

namespace lifuren {

// 损失函数
template<typename T>
concept L = std::derived_from<T, torch::nn::Module>;

// 模型结构
template<typename T>
concept M = std::derived_from<T, torch::nn::Module>;

/**
 * 模型训练器
 * 
 * @param P 优化函数
 * @param M 模型结构
 * @param D 数据集
 * 
 * @author acgist
 */
template<typename P, typename M, typename D>
class Trainer {

protected:
    lifuren::config::ModelParams params{}; // 模型参数
    D trainDataset{ nullptr }; // 训练数据集
    D valDataset  { nullptr }; // 验证数据集
    D testDataset { nullptr }; // 测试数据集
    M model       { nullptr }; // 模型实现
    std::unique_ptr<P> optimizer{ nullptr }; // 优化函数

public:
    torch::DeviceType device{ torch::DeviceType::CPU }; // 计算设备

public:
    /**
     * @param params 模型参数
     */
    Trainer(lifuren::config::ModelParams params = {});
    virtual ~Trainer();

public:
    // 保存模型
    virtual bool save(const std::string& path = "./lifuren.pt");
    // 加载模型
    virtual bool load(const std::string& path = "./lifuren.pt", const bool print_model = true);
    // 定义模型
    virtual bool define(const bool define_weight = true, const bool define_dataset = true, const bool define_optimizer = true, const bool print_model = true);
    // 打印模型
    virtual void print();
    // 训练模型
    virtual void trainValAndTest(const bool val = true, const bool test = true);
    
protected:
    // 初始化权重
    virtual void defineWeight();
    // 定义数据集
    virtual void defineDataset() = 0;
    // 定义优化函数
    virtual void defineOptimizer();
    // 计算损失
    virtual void loss(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) = 0;
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
     * @param duration 时间消耗
     */
    virtual void evaluation(
        const char*  name,
        const size_t epoch,
        const float  loss,
        const size_t duration
    );

};

} // END OF lifuren

template<typename P, typename M, typename D>
lifuren::Trainer<P, M, D>::Trainer(lifuren::config::ModelParams params) : params(std::move(params)), device(lifuren::get_device()) {
    this->model = M{ this->params };
    if(this->params.thread_size == 0) {
        this->params.thread_size = std::thread::hardware_concurrency();
    }
    torch::set_num_threads(this->params.thread_size);
    SPDLOG_DEBUG("定义模型：{}", this->params.model_name);
    SPDLOG_DEBUG("计算设备：{}", torch::DeviceTypeName(this->device));
}

template<typename P, typename M, typename D>
lifuren::Trainer<P, M, D>::~Trainer() {
    SPDLOG_DEBUG("释放模型：{}", this->params.model_name);
}

template<typename P, typename M, typename D>
bool lifuren::Trainer<P, M, D>::save(const std::string& path) {
    if(!this->model) {
        SPDLOG_WARN("模型保存失败：没有定义模型");
        return false;
    }
    lifuren::file::create_parent_directory(path);
    this->model->eval();
    this->model->to(torch::DeviceType::CPU);
    torch::save(model, path);
    this->model->to(this->device);
    SPDLOG_INFO("保存模型：{}", path);
    return true;
}

template<typename P, typename M, typename D>
bool lifuren::Trainer<P, M, D>::load(const std::string& path, const bool print_model) {
    if(!lifuren::file::is_file(path)) {
        SPDLOG_WARN("加载模型失败：{}", path);
        return false;
    }
    SPDLOG_INFO("加载模型：{}", path);
    try {
        torch::load(this->model, path, torch::DeviceType::CPU);
    } catch(const std::exception& e) {
        SPDLOG_ERROR("加载模型异常：{} - {}", path, e.what());
        return false;
    }
    this->model->to(this->device);
    this->model->eval();
    if(print_model) {
        this->print();
    }
    return true;
}

template<typename P, typename M, typename D>
bool lifuren::Trainer<P, M, D>::define(const bool define_weight, const bool define_dataset, const bool define_optimizer, const bool print_model) {
    if(define_weight) {
        this->defineWeight();
    }
    if(define_dataset) {
        this->defineDataset();
    }
    if(define_optimizer) {
        this->defineOptimizer();
    }
    this->model->to(this->device);
    this->model->eval();
    if(print_model) {
        this->print();
    }
    return true;
}

template<typename P, typename M, typename D>
inline void lifuren::Trainer<P, M, D>::print() {
    size_t total_numel = 0;
    for(const auto& parameter : this->model->named_parameters()) {
        total_numel += parameter.value().numel();
        SPDLOG_DEBUG("模型参数数量: {} = {}", parameter.key(), parameter.value().numel());
    }
    SPDLOG_DEBUG("模型参数总量: {}", total_numel);
}

template<typename P, typename M, typename D>
void lifuren::Trainer<P, M, D>::trainValAndTest(const bool val, const bool test) {
    if(!this->trainDataset) {
        SPDLOG_WARN("无效的训练数据集");
        return;
    }
    SPDLOG_INFO("开始训练：{}", this->params.model_name);
    const auto a = std::chrono::system_clock::now();
    try {
        auto scheduler = torch::optim::StepLR(*this->optimizer, 3, 0.999);
        for (size_t epoch = 1; epoch <= this->params.epoch_size; ++epoch) {
            this->train(epoch);
            scheduler.step();
            if(val) {
                this->val(epoch);
            }
            if(this->params.check_point && epoch % this->params.check_epoch == 1) {
                this->save(lifuren::file::join({
                    this->params.model_path,
                    this->params.model_name + "." + std::to_string(epoch) + ".ckpt"
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

template<typename P, typename M, typename D>
inline void lifuren::Trainer<P, M, D>::defineWeight() {
}

template<typename P, typename M, typename D>
inline void lifuren::Trainer<P, M, D>::defineOptimizer() {
    this->optimizer = std::make_unique<P>(this->model->parameters(), this->params.lr);
}

template<typename P, typename M, typename D>
void lifuren::Trainer<P, M, D>::train(const size_t epoch) {
    if(!this->trainDataset) {
        SPDLOG_WARN("无效的训练数据集");
        return;
    }
    this->model->train();
    double loss_val    = 0.0;
    size_t batch_count = 0;
    const auto a = std::chrono::system_clock::now();
    for (const auto& batch : *this->trainDataset) {
        torch::Tensor pred;
        torch::Tensor loss;
        torch::Tensor data   = batch.data;
        torch::Tensor target = batch.target;
        this->optimizer->zero_grad();
        this->loss(data, target, pred, loss);
        loss.backward();
        if(this->params.grad_clip > 0.0F) {
            torch::nn::utils::clip_grad_norm_(this->model->parameters(), this->params.grad_clip);
        }
        this->optimizer->step();
        loss_val += loss.template item<float>();
        ++batch_count;
    }
    const auto z = std::chrono::system_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count();
    this->evaluation("训练", epoch, loss_val / batch_count, duration);
}

template<typename P, typename M, typename D>
void lifuren::Trainer<P, M, D>::val(const size_t epoch) {
    if(!this->valDataset) {
        return;
    }
    torch::NoGradGuard no_grad_guard;
    this->model->eval();
    double loss_val    = 0.0;
    size_t batch_count = 0;
    const auto a = std::chrono::system_clock::now();
    for (const auto& batch : *this->valDataset) {
        torch::Tensor pred;
        torch::Tensor loss;
        torch::Tensor data   = batch.data;
        torch::Tensor target = batch.target;
        this->loss(data, target, pred, loss);
        loss_val += loss.template item<float>();
        ++batch_count;
    }
    const auto z = std::chrono::system_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count();
    this->evaluation("验证", epoch, loss_val / batch_count, duration);
}

template<typename P, typename M, typename D>
void lifuren::Trainer<P, M, D>::test() {
    if(!this->testDataset) {
        return;
    }
    torch::NoGradGuard no_grad_guard;
    this->model->eval();
    double loss_val    = 0.0;
    size_t batch_count = 0;
    const auto a = std::chrono::system_clock::now();
    for (const auto& batch : *this->testDataset) {
        torch::Tensor pred;
        torch::Tensor loss;
        torch::Tensor data   = batch.data;
        torch::Tensor target = batch.target;
        this->loss(data, target, pred, loss);
        loss_val += loss.template item<float>();
        ++batch_count;
    }
    const auto z = std::chrono::system_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(z - a).count();
    this->evaluation("测试", 0, loss_val / batch_count, duration);
}

template<typename P, typename M, typename D>
inline void lifuren::Trainer<P, M, D>::evaluation(
    const char*  name,
    const size_t epoch,
    const float  loss,
    const size_t duration
) {
    SPDLOG_INFO(
        "当前{}第 {} 轮，损失值为：{:.6f}，耗时：{}。",
        name,
        epoch,
        loss,
        duration
    );
}

#endif // END OF LFR_HEADER_CORE_TRAINER_HPP
