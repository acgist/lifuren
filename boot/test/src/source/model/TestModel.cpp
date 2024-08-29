#include "lifuren/Model.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

class SimpleModel : public lifuren::Model<int, int> {

private:
    ggml_tensor* fc1_weight{ nullptr };
    ggml_tensor* fc1_bias  { nullptr };

public:
    SimpleModel() : Model() {
        this->names.reserve(2);
        this->names.push_back("fc1.weight");
        this->names.push_back("fc1.bias");
        this->datas  = ggml_new_tensor_1d(this->ctx_compute, GGML_TYPE_F32, 100);
        this->labels = ggml_new_tensor_1d(this->ctx_compute, GGML_TYPE_F32, 1);
    }
    ~SimpleModel() {
    }

public:
    Model& bind() override {
        if(this->weights.empty()) {
            SPDLOG_WARN("绑定权重失败：权重为空");
            return *this;
        }
        this->fc1_weight = this->weights["fc1.weight"];
        this->fc1_bias   = this->weights["fc1.bias"];
        return *this;
    };
    // 初始化模型
    Model& init() override {
        this->fc1_weight = ggml_new_tensor_1d(this->ctx_weight, this->type, 100);
        this->fc1_bias   = ggml_new_tensor_1d(this->ctx_weight, this->type, 1);
        this->weights.emplace("fc1.weight", this->fc1_weight);
        this->weights.emplace("fc1.bias",   this->fc1_bias);
        return *this;
    };
    ggml_tensor* buildLoss() override {
        // return ggml_cross_entropy_loss(this->ctx_compute, this->logits, this->labels);
        ggml_tensor* result = ggml_new_tensor_1d(this->ctx_compute, this->type, this->batch_size);
        float* preds  = ggml_get_data_f32(this->logits);
        float* labels = ggml_get_data_f32(this->labels);
        float* data   = ggml_get_data_f32(result);
        for(int index = 0; index < this->batch_size; ++index) {
            data[index] = preds[index] - labels[index];
        }
        return result;
    };
    // 创建计算图
    ggml_tensor* buildLogits() override {
        return ggml_relu(this->ctx_compute, 
            ggml_add(this->ctx_compute,
                ggml_mul_mat(this->ctx_compute, this->fc1_weight, this->datas),
                this->fc1_bias
            )
        );
    };

};

static void testSaveLoad() {
    SimpleModel save{};
    save.init().rand().build().print().save("D:/tmp");
    SimpleModel load{};
    load.load("D:/tmp").bind().build().print();
}

static void testSaveLoadWeight() {
    SimpleModel save{};
    save.init().rand().build().print().saveWeight("D:/tmp");
    SimpleModel load{};
    load.loadWeight("D:/tmp").print();
}

static void testLine() {
    std::random_device device;
    std::mt19937 rand(device());
    std::normal_distribution<> weight(10, 2);
    std::normal_distribution<> bias  (0 , 2);
    float features[1000];
    float labels  [1000];
    for(int index = 0; index < 1000; ++index) {
        features[index] = weight(rand);
        labels  [index] = features[index] * 15.4 + 4 + bias(rand);
    }
    lifuren::datasets::TensorDataset dataset{
        1000,
        10,
        features,
        1,
        labels,
        1
    };
    SimpleModel save{};
    save.init().rand().build().print();
    save.trainDataset.reset(&dataset);
    save.trainAndVal();
    float data[] { 3.2 };
    float* pred = save.eval(data, 1);
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    // testSaveLoad();
    testSaveLoadWeight();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}