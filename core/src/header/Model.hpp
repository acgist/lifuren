/**
 * 模型
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_MODEL_HPP
#define LFR_HEADER_CORE_MODEL_HPP

#include "torch/torch.h"
#include "torch/script.h"

#include "./config/Config.hpp"

namespace lifuren {

/**
 * 李夫人模型
 * 
 * @param M 模型配置
 * @param I 预测输入
 * @param R 预测输出
 * 
 * @author acgist
 */
template <typename M, typename I, typename R>
class Model {

static_assert(std::is_base_of_v<lifuren::ModelConfig, M>, "必须继承模型配置");

public:
    // 基本配置
    lifuren::Config config;
    // 模型配置
    M modelConfig;

public:
    Model();
    virtual ~Model();
    /**
     * @param config      基本配置
     * @param modelConfig 模型配置
     */
    Model(const lifuren::Config& config, const M& modelConfig);

public:
    // 保存模型
    virtual void save();
    // 加载模型
    virtual void load();
    // 训练模型
    virtual void train() = 0;
    // 验证模型
    virtual void val() = 0;
    // 测试模型
    virtual void test() = 0;
    /**
     * 模型预测
     * 
     * @param input 预测输入
     * 
     * @return 预测输出
     */
    virtual R pred(I input) = 0;
    // 训练验证
    virtual void trainAndVal();

};

template<typename M, typename I, typename R>
lifuren::Model<M, I, R>::Model() {
}

template<typename M, typename I, typename R>
lifuren::Model<M, I, R>::~Model() {
}

template<typename M, typename I, typename R>
lifuren::Model<M, I, R>::Model(const lifuren::Config& config, const M& modelConfig) : config(config), modelConfig(modelConfig) {
}

template<typename M, typename I, typename R>
void lifuren::Model<M, I, R>::save() {
}

template<typename M, typename I, typename R>
void lifuren::Model<M, I, R>::load() {
}

template<typename M, typename I, typename R>
void lifuren::Model<M, I, R>::trainAndVal() {
    //     SPDLOG_DEBUG("是否使用CUDA：{}", this->device.is_cuda());
    // std::filesystem::path data_path = data_dir;
    // std::string path_val   = (data_path / "val").u8string();
    // std::string path_train = (data_path / "train").u8string();
    // std::map<std::string, int> mapping = {
    //     { "man"  , 1 },
    //     { "woman", 0 }
    // };
    // this->model->to(this->device);
    // auto data_loader_val   = lifuren::datasets::loadImageFileDataset(200, 200, batch_size, path_val,   image_type, mapping);
    // auto data_loader_train = lifuren::datasets::loadImageFileDataset(200, 200, batch_size, path_train, image_type, mapping);
    // for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
    //     if (epoch == num_epochs / 4) {
    //         learning_rate /= 10;
    //     }
    //     try {
    //         torch::optim::Adam optimizer(this->model->parameters(), learning_rate);
    //         auto a = std::chrono::system_clock::now();
    //         this->trian(epoch, batch_size, optimizer, data_loader_train);
    //         auto z = std::chrono::system_clock::now();
    //         SPDLOG_DEBUG("训练耗时：{}", std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count());
    //         this->val(epoch, batch_size, data_loader_val);
    //     } catch(const std::exception& e) {
    //         SPDLOG_ERROR("训练异常：{}", e.what());
    //     }
    // }
    // torch::save(this->model, save_path);
}

}

#endif // LFR_HEADER_CORE_MODEL_HPP
