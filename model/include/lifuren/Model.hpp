/**
 * 模型
 * 
 * 模型输出向量不要使用任何逻辑判断语句
 * 
 * @author acgist
 * 
 * TODO: GPU
 * 
 * https://pytorch.org/cppdocs/
 */
#ifndef LFR_HEADER_MODEL_MODEL_HPP
#define LFR_HEADER_MODEL_MODEL_HPP

#include <memory>
#include <string>
#include <thread>
#include <concepts>

#include "torch/nn.h"
#include "torch/jit.h"
#include "torch/optim.h"
#include "torch/serialize.h"

#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Logger.hpp"

LFR_FORMAT_LOG_STREAM(at::Tensor)
LFR_FORMAT_LOG_STREAM(c10::IntArrayRef)

namespace lifuren {

/**
 * 模型参数
 */
struct ModelParams {

    float       lr         { 0.001F    }; // 学习率
    size_t      batch_size { 100LL     }; // 批量大小
    size_t      epoch_count{ 128LL     }; // 训练次数
    bool        classify   { false     }; // 分类任务
    bool        check_point{ false     }; // 保存快照
    std::string check_path { "./"      }; // 快照路径
    std::string model_name { "lifuren" }; // 模型名称
    std::string train_path {}; // 训练数据集路径
    std::string val_path   {}; // 验证数据集路径
    std::string test_path  {}; // 测试数据集路径
    torch::DeviceType device{ torch::DeviceType::CPU }; // 计算设备
    size_t thread_size{ std::thread::hardware_concurrency() }; // 线程数量

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
 * @param L 损失函数
 * @param P 优化函数
 * @param M 模型结构
 * 
 * @author acgist
 */
template<typename D, typename L, typename P, typename M>
class Model {

protected:
    ModelParams params{};      // 模型参数
    D trainDataset{ nullptr }; // 训练数据集
    D valDataset  { nullptr }; // 验证数据集
    D testDataset { nullptr }; // 测试数据集
    L loss        { nullptr }; // 损失函数
    M model       { nullptr }; // 模型结构
    std::unique_ptr<P> optimizer{ nullptr }; // 优化函数

public:
    Model(
        ModelParams params = {},
        L loss  = {},
        M model = {}
    );
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
    virtual void train(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss);
    // 验证模型
    virtual void val(size_t epoch);
    virtual void val(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss);
    // 测试模型
    virtual void test();
    virtual void test(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss);
    // 计算逻辑
    virtual void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss);
    // 训练验证测试模型
    virtual void trainValAndTest(const bool& val = true, const bool& test = true);
    // 模型预测
    virtual torch::Tensor pred(const torch::Tensor& input);

protected:
    // 定义数据集
    virtual bool defineDataset() = 0;

};

}

template<typename D, typename L, typename P, typename M>
lifuren::Model<D, L, P, M>::Model(
    lifuren::ModelParams params,
    L loss,
    M model
) : params(std::move(params)),
    loss(std::move(loss)),
    model(std::move(model))
{
    if(this->model) {
        this->optimizer = std::make_unique<P>(this->model->parameters(), this->params.lr);
    }
    lifuren::setDevice(this->params.device);
    torch::set_num_threads(this->params.thread_size);
    // torch::set_default_dtype(this->params.device);
    SPDLOG_DEBUG("当前计算设备：{}", torch::DeviceTypeName(this->params.device));
}

template<typename D, typename L, typename P, typename M>
lifuren::Model<D, L, P, M>::~Model() {
}

template<typename D, typename L, typename P, typename M>
bool lifuren::Model<D, L, P, M>::save(const std::string& path, const std::string& filename) {
    if(!this->model) {
        SPDLOG_WARN("保存模型没有定义");
        return false;
    }
    const std::string fullpath = lifuren::file::join({ path, filename }).string();
    SPDLOG_DEBUG("保存模型：{}", fullpath);
    this->model->eval();
    torch::save(this->model, fullpath);
    return true;
}

template<typename D, typename L, typename P, typename M>
bool lifuren::Model<D, L, P, M>::load(const std::string& path, const std::string& filename) {
    const std::string fullpath = lifuren::file::join({ path, filename }).string();
    if(!lifuren::file::exists(fullpath)) {
        SPDLOG_WARN("模型文件无效：{}", fullpath);
        return false;
    }
    SPDLOG_DEBUG("加载模型：{}", fullpath);
    torch::load(this->model, fullpath);
    return true;
}

template<typename D, typename L, typename P, typename M>
bool lifuren::Model<D, L, P, M>::define() {
    this->model->to(this->params.device);
    return this->defineDataset();
}

template<typename D, typename L, typename P, typename M>
void lifuren::Model<D, L, P, M>::print() {
    for(const auto& value : this->model->parameters()) {
        SPDLOG_DEBUG("parameters: {}", value);
    }
    for(const auto& value : this->model->named_parameters()) {
        SPDLOG_DEBUG("named_parameters: {} = {}", value.key(), value.value());
    }
}

template<typename D, typename L, typename P, typename M>
void lifuren::Model<D, L, P, M>::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    pred = this->model->forward(feature);
    loss = this->loss(pred, label);
}

template<typename D, typename L, typename P, typename M>
void lifuren::Model<D, L, P, M>::train(size_t epoch) {
    if(!this->trainDataset) {
        SPDLOG_WARN("无效的训练数据集");
        return;
    }
    size_t accu_val = 0;
    size_t data_val = 0;
    double loss_val = 0.0;
    size_t batch_count = 0;
    this->model->train();
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
            auto accu = pred.argmax(1).eq(target).sum();
            accu_val += accu.template item<int>();
            data_val += target.numel();
        }
        loss_val += loss.template item<float>();
        ++batch_count;
    }
    const auto z = std::chrono::system_clock::now();
    if(this->params.classify) {
        SPDLOG_INFO(
            "当前训练第 {} 轮，损失值为：{:.6f}，正确率为：{} / {}，耗时：{}。",
            epoch + 1,
            loss_val / batch_count,
            accu_val,
            data_val,
            std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count()
        );
    } else {
        SPDLOG_INFO(
            "当前训练第 {} 轮，损失值为：{:.6f}，耗时：{}。",
            epoch + 1,
            loss_val / batch_count,
            std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count()
        );
    }
}

template<typename D, typename L, typename P, typename M>
void lifuren::Model<D, L, P, M>::train(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    this->logic(feature, label, pred, loss);
}

template<typename D, typename L, typename P, typename M>
void lifuren::Model<D, L, P, M>::val(size_t epoch) {
    if(!this->valDataset) {
        return;
    }
    size_t accu_val = 0;
    size_t data_val = 0;
    double loss_val = 0.0;
    size_t batch_count = 0;
    this->model->eval();
    const auto a = std::chrono::system_clock::now();
    for (auto& batch : *this->valDataset) {
        torch::Tensor pred;
        torch::Tensor loss;
        torch::Tensor data   = batch.data;
        torch::Tensor target = batch.target;
        this->val(data, target, pred, loss);
        if(this->params.classify) {
            auto accu = pred.argmax(1).eq(target).sum();
            accu_val += accu.template item<int>();
            data_val += target.numel();
        }
        loss_val += loss.template item<float>();
        ++batch_count;
    }
    const auto z = std::chrono::system_clock::now();
    if(this->params.classify) {
        SPDLOG_INFO(
            "当前验证第 {} 轮，损失值为：{:.6f}，正确率为：{} / {}，耗时：{}。",
            epoch + 1,
            loss_val / batch_count,
            accu_val,
            data_val,
            std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count()
        );
    } else {
        SPDLOG_INFO(
            "当前验证第 {} 轮，损失值为：{:.6f}，耗时：{}。",
            epoch + 1,
            loss_val / batch_count,
            std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count()
        );
    }
}

template<typename D, typename L, typename P, typename M>
void lifuren::Model<D, L, P, M>::val(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
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
    const auto a = std::chrono::system_clock::now();
    for (auto& batch : *this->testDataset) {
        torch::Tensor pred;
        torch::Tensor loss;
        torch::Tensor data   = batch.data;
        torch::Tensor target = batch.target;
        this->test(data, target, pred, loss);
        if(this->params.classify) {
            auto accu = pred.argmax(1).eq(target).sum();
            accu_val += accu.template item<int>();
            data_val += target.numel();
        }
        loss_val += loss.template item<float>();
        ++batch_count;
    }
    const auto z = std::chrono::system_clock::now();
    if(this->params.classify) {
        SPDLOG_INFO(
            "当前测试损失值为：{:.6f}，正确率为：{} / {}，耗时：{}。",
            loss_val / batch_count,
            accu_val,
            data_val,
            std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count()
        );
    } else {
        SPDLOG_INFO(
            "当前测试损失值为：{:.6f}，耗时：{}。",
            loss_val / batch_count,
            std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count()
        );
    }
}

template<typename D, typename L, typename P, typename M>
void lifuren::Model<D, L, P, M>::test(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    this->logic(feature, label, pred, loss);
}

template<typename D, typename L, typename P, typename M>
void lifuren::Model<D, L, P, M>::trainValAndTest(const bool& val, const bool& test) {
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
                this->save(this->params.check_path, this->params.model_name + "-" + std::to_string(epoch + 1) + ".pt");
            }
        }
        if(test) {
            this->test();
        }
    } catch(const std::exception& e) {
        SPDLOG_ERROR("训练异常：{}", e.what());
    }
    const auto z = std::chrono::system_clock::now();
    SPDLOG_DEBUG("累计耗时：{}", std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count());
}

template<typename D, typename L, typename P, typename M>
torch::Tensor lifuren::Model<D, L, P, M>::pred(const torch::Tensor& input) {
    this->model->eval();
    return this->model->forward(input);
}

#endif // LFR_HEADER_MODEL_MODEL_HPP
