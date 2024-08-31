#include "lifuren/Model.hpp"

#include <random>

#include "ggml.h"

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

#include "lifuren/Datasets.hpp"

class SimpleModel : public lifuren::Model {

private:
    ggml_tensor* fc1_weight{ nullptr };
    ggml_tensor* fc1_bias  { nullptr };

public:
    SimpleModel(lifuren::Model::ModelParams params = {}) : Model(params) {
        this->names.reserve(2);
        this->names.push_back("fc1.weight");
        this->names.push_back("fc1.bias");
    }
    ~SimpleModel() {
    }

public:
    Model& bindWeight() override {
        if(this->weights.empty()) {
            SPDLOG_WARN("绑定权重失败：权重为空");
            return *this;
        }
        this->fc1_weight = this->weights["fc1.weight"];
        this->fc1_bias   = this->weights["fc1.bias"];
        return *this;
    };
    // 初始化模型
    Model& defineWeight() override {
        this->fc1_weight = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, 100, 1);
        this->fc1_bias   = ggml_new_tensor_2d(this->ctx_weight, GGML_TYPE_F32, 100, 1);
        this->weights.emplace("fc1.weight", this->fc1_weight);
        this->weights.emplace("fc1.bias",   this->fc1_bias);
        return *this;
    };
    ggml_tensor* buildDatas() override {
        return ggml_new_tensor_2d(this->ctx_compute, GGML_TYPE_F32, 100, 1);
    }
    ggml_tensor* buildLabels() override {
        return ggml_new_tensor_2d(this->ctx_compute, GGML_TYPE_F32, 100, 1);
    }
    ggml_tensor* buildLoss() override {
        return ggml_cross_entropy_loss(this->ctx_compute, this->logits, this->labels);
        return ggml_sub(this->ctx_compute, this->logits, this->labels);
        // auto result = ggml_sub(this->ctx_compute, this->logits, this->labels);
        // bool is_node = false;
        // if (this->logits->grad || this->labels->grad) {
        //     is_node = true;
        // }
        // result->ne[1] = 1;
        // result->ne[2] = 1;
        // result->ne[3] = 1;
        // result->grad = is_node ? ggml_dup_tensor(this->ctx_compute, result) : NULL;
        // result->src[0] = this->logits;
        // result->src[1] = this->labels;
        // return result;
        // ggml_tensor* result = ggml_new_tensor_1d(this->ctx_compute, GGML_TYPE_F32, 1);
        // float* preds  = ggml_get_data_f32(this->logits);
        // float* labels = ggml_get_data_f32(this->labels);
        // float* data   = ggml_get_data_f32(result);
        // double sum = 0.0;
        // for(int index = 0; index < this->params.batch_size; ++index) {
        //     sum = sum + std::abs(preds[index] - labels[index]);
        // }
        // *data = sum / this->params.batch_size;
        // bool is_node = false;
        // if (this->logits->grad || this->labels->grad) {
        //     is_node = true;
        // }
        // result->grad = is_node ? ggml_dup_tensor(this->ctx_compute, result) : NULL;
        // result->src[0] = this->logits;
        // result->src[1] = this->labels;
        // return result;
    };
    // 创建计算图
    ggml_tensor* buildLogits() override {
        return 
            ggml_add(this->ctx_compute,
                ggml_mul(this->ctx_compute, this->fc1_weight, this->datas),
                this->fc1_bias
            );
    };

};

static void testSaveLoad() {
    SimpleModel save{};
    // save.define().print().save("D:/tmp");
    save.define().print().saveEval("D:/tmp");
    SimpleModel load{};
    // load.load("D:/tmp").print();
    load.loadEval("D:/tmp").print();
}

static void testLine() {
    std::random_device device;
    std::mt19937 rand(device());
    std::normal_distribution<> weight(10, 2);
    std::normal_distribution<> bias  (0 , 2);
    float features[210];
    float labels  [210];
    for(int index = 0; index < 210; ++index) {
        features[index] = weight(rand);
        labels  [index] = features[index] * 15.4 + 4 + bias(rand);
    }
    for(int i = 0; i < 210; ++i) {
        auto v = features[i];
        SPDLOG_DEBUG("d = {}", v);
    }
    for(int i = 0; i < 210; ++i) {
        auto v = labels[i];
        SPDLOG_DEBUG("l = {}", v);
    }
    lifuren::datasets::TensorDataset* dataset = new lifuren::datasets::TensorDataset{
        210,
        100,
        features,
        1,
        labels,
        1
    };
    lifuren::Model::ModelParams params {
        .epoch_count = 1
    };
    SimpleModel save{params};
    save.define();
    // save.print();
    save.trainDataset.reset(dataset);
    save.trainAndVal();
    float data[] { 3.2 };
    float* pred = save.eval(data, 1);
    SPDLOG_DEBUG("当前预测：{}", *pred);
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testLine();
    // testSaveLoad();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}