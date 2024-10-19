/**
 * 模型
 * 
 * TODO:
 * 1. CPU/GPU = model or dataset .to(this->device).to(torch::kF32)
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_MODEL_MODEL_HPP
#define LFR_HEADER_MODEL_MODEL_HPP

#include <memory>
#include <string>
#include <thread>
#include <concepts>

#include "torch/torch.h"
#include "torch/script.h"

#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"

#include "lifuren/File.hpp"
#include "lifuren/Logger.hpp"

LFR_FORMAT_LOG_STREAM(at::Tensor)

namespace lifuren {

/**
 * 模型参数
 */
struct ModelParams {

    int8_t      device     { 0      }; // 计算设备：torch::DeviceType::CPU
    float       lr         { 0.001F }; // 学习率
    size_t      batch_size { 100LL  }; // 批量大小
    size_t      epoch_count{ 128LL  }; // 训练次数
    bool        classify   { false  }; // 分类任务
    bool        check_point{ false  }; // 保存快照
    size_t      check_index{ 0LL    }; // 快照索引
    std::string check_path { "./"   }; // 快照路径
    size_t      thread_size{ std::thread::hardware_concurrency() }; // 线程数量

};

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
 * @param O 模型输出
 * @param I 模型输入
 * @param L 损失函数
 * @param M 模型结构
 * 
 * @author acgist
 */
template<typename D, typename O, typename I, typename L, typename M>
class Model {

protected:
    ModelParams params{};      // 模型参数
    D trainDataset{ nullptr }; // 训练数据集
    D valDataset  { nullptr }; // 验证数据集
    D testDataset { nullptr }; // 测试数据集
    L loss        { nullptr }; // 损失函数
    M model       { nullptr }; // 模型结构
    std::shared_ptr<torch::optim::Optimizer> optimizer{ nullptr }; // 优化函数

public:
    Model(L loss, M model, ModelParams params = {});
    virtual ~Model();

public:
    // 模型保存
    virtual bool save(const std::string& path = "./", const std::string& filename = "lifuren.pt");
    // 模型加载
    virtual bool load(const std::string& path = "./", const std::string& filename = "lifuren.pt");
    // 定义模型
    virtual bool define();
    // 打印模型
    virtual void print();
    // 训练模型
    virtual void train(size_t epoch);
    // 验证模型
    virtual void val(size_t epoch);
    // 测试模型
    virtual void test();
    // 训练验证测试模型
    virtual void trainValAndTest(const bool val = true, const bool test = true);
    // 模型预测
    virtual O eval(I i) = 0;

protected:
    // 定义数据集
    virtual bool defineDataset() = 0;
    // 定义优化函数
    virtual std::shared_ptr<torch::optim::Optimizer> defineOptimizer() = 0;

};

}

template<typename D, typename O, typename I, typename L, typename M>
lifuren::Model<D, O, I, L, M>::Model(
    L loss,
    M model,
    lifuren::ModelParams params
) : loss(loss),
    model(model),
    params(params)
{
}

template<typename D, typename O, typename I, typename L, typename M>
lifuren::Model<D, O, I, L, M>::~Model() {
}

template<typename D, typename O, typename I, typename L, typename M>
bool lifuren::Model<D, O, I, L, M>::save(const std::string& path, const std::string& filename) {
    if(!this->model) {
        SPDLOG_WARN("保存模型没有定义");
        return false;
    }
    const std::string fullpath = lifuren::file::join({ path, filename }).string();
    SPDLOG_DEBUG("保存模型：{}", fullpath);
    torch::save(this->model, fullpath);
    return true;
}

template<typename D, typename O, typename I, typename L, typename M>
bool lifuren::Model<D, O, I, L, M>::load(const std::string& path, const std::string& filename) {
    const std::string fullpath = lifuren::file::join({ path, filename }).string();
    SPDLOG_DEBUG("加载模型：{}", fullpath);
    torch::load(this->model, fullpath);
    return true;
}

template<typename D, typename O, typename I, typename L, typename M>
bool lifuren::Model<D, O, I, L, M>::define() {
    if(this->defineDataset()) {
        this->optimizer = this->defineOptimizer();
        return true;
    }
    return false;
}

template<typename D, typename O, typename I, typename L, typename M>
void lifuren::Model<D, O, I, L, M>::print() {
    // for(const auto& value : this->model->modules()) {
    //     SPDLOG_DEBUG("modules: {}", value);
    // }
    // for(const auto& value : this->model->named_modules()) {
    //     SPDLOG_DEBUG("named_modules: {} = {}", value.key(), value.value());
    // }
    for(const auto& value : this->model->parameters()) {
        SPDLOG_DEBUG("parameters: {}", value);
    }
    for(const auto& value : this->model->named_parameters()) {
        SPDLOG_DEBUG("named_parameters: {} = {}", value.key(), value.value());
    }
}

template<typename D, typename O, typename I, typename L, typename M>
void lifuren::Model<D, O, I, L, M>::train(size_t epoch) {
    if(!this->trainDataset) {
        SPDLOG_WARN("无效的训练数据集");
        return;
    }
    this->model->train();
    double accu_val = 0.0;
    double loss_val = 0.0;
    size_t count = 0LL;
    size_t batch_count = 0LL;
    auto a = std::chrono::system_clock::now();
    for (const auto& batch : *this->trainDataset) {
        auto data   = batch.data;
        auto target = batch.target;
        torch::Tensor pred = this->model->forward(data);
        torch::Tensor loss = this->loss->forward(pred, target);
        this->optimizer->zero_grad();
        loss.backward();
        this->optimizer->step();
        if(this->params.classify) {
            auto accu = pred.argmax(1).eq(target).sum();
            accu_val += accu.template item<float>();
        }
        loss_val += loss.template item<float>();
        ++batch_count;
        count += this->trainDataset->options().batch_size;
    }
    auto z = std::chrono::system_clock::now();
    if(this->params.classify) {
        SPDLOG_INFO(
            "当前训练第 {} 轮，损失值为：{}，正确率为：{} / {}，耗时：{}。",
            epoch,
            loss_val / batch_count,
            accu_val,
            count,
            std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count()
        );
    } else {
        SPDLOG_INFO(
            "当前训练第 {} 轮，损失值为：{}，耗时：{}。",
            epoch,
            loss_val / batch_count,
            std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count()
        );
    }
}

template<typename D, typename O, typename I, typename L, typename M>
void lifuren::Model<D, O, I, L, M>::val(size_t epoch) {
    if(!this->valDataset) {
        SPDLOG_WARN("无效的验证数据集");
        return;
    }
    this->model->eval();
    double accu_val = 0.0;
    double loss_val = 0.0;
    size_t count = 0LL;
    size_t batch_count = 0LL;
    auto a = std::chrono::system_clock::now();
    for (auto& batch : *this->valDataset) {
        auto data   = batch.data;
        auto target = batch.target;
        torch::Tensor pred = this->model->forward(data);
        torch::Tensor loss = this->loss->forward(pred, target);
        if(this->params.classify) {
            auto accu = pred.argmax(1).eq(target).sum();
            accu_val += accu.template item<float>();
        }
        loss_val += loss.template item<float>();
        ++batch_count;
        count += this->trainDataset->options().batch_size;
    }
    auto z = std::chrono::system_clock::now();
    if(this->params.classify) {
        SPDLOG_INFO(
            "当前验证第 {} 轮，损失值为：{}，正确率为：{} / {}，耗时：{}。",
            epoch,
            loss_val / batch_count,
            accu_val,
            count,
            std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count()
        );
    } else {
        SPDLOG_INFO(
            "当前验证第 {} 轮，损失值为：{}，耗时：{}。",
            epoch,
            loss_val / batch_count,
            std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count()
        );
    }
}

template<typename D, typename O, typename I, typename L, typename M>
void lifuren::Model<D, O, I, L, M>::test() {
    if(!this->testDataset) {
        SPDLOG_WARN("无效的测试数据集");
        return;
    }
    this->model->eval();
    double accu_val = 0.0;
    double loss_val = 0.0;
    size_t count = 0LL;
    size_t batch_count = 0LL;
    auto a = std::chrono::system_clock::now();
    for (auto& batch : *this->testDataset) {
        auto data   = batch.data;
        auto target = batch.target;
        torch::Tensor pred = this->model->forward(data);
        torch::Tensor loss = this->loss->forward(pred, target);
        if(this->params.classify) {
            auto accu = pred.argmax(1).eq(target).sum();
            accu_val += accu.template item<float>();
        }
        loss_val += loss.template item<float>();
        ++batch_count;
        count += this->trainDataset->options().batch_size;
    }
    auto z = std::chrono::system_clock::now();
    if(this->params.classify) {
        SPDLOG_INFO(
            "当前测试损失值为：{}，正确率为：{} / {}，耗时：{}。",
            loss_val / batch_count,
            accu_val,
            count,
            std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count()
        );
    } else {
        SPDLOG_INFO(
            "当前测试损失值为：{}，耗时：{}。",
            loss_val / batch_count,
            std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count()
        );
    }
}

template<typename D, typename O, typename I, typename L, typename M>
void lifuren::Model<D, O, I, L, M>::trainValAndTest(const bool val, const bool test) {
    auto a = std::chrono::system_clock::now();
    for (size_t epoch = 0LL; epoch < this->params.epoch_count; ++epoch) {
        this->train(epoch);
        if(val) {
            this->val(epoch);
        }
    }
    if(test) {
        this->test();
    }
    auto z = std::chrono::system_clock::now();
    SPDLOG_DEBUG("累计耗时：{}", std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count());
}

#endif // LFR_HEADER_MODEL_MODEL_HPP
