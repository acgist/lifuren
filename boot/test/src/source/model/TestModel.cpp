#include "lifuren/Model.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/Logger.hpp"

class SimpleModel : public lifuren::Model<int, int> {

private:
    ggml_tensor* fc1_weight;
    ggml_tensor* fc1_bias;

public:
    SimpleModel() : Model() {
        this->names.reserve(2);
        this->names.push_back("fc1.weight");
        this->names.push_back("fc1.bias");
    }
    ~SimpleModel() {
    }

public:
    Model& bind() override {
        if(this->weights.empty()) {
            return *this;
        }
        this->fc1_weight = this->weights["fc1.weight"];
        this->fc1_bias   = this->weights["fc1.bias"];
        return *this;
    };
    // 初始化模型
    Model& init() override {
        this->fc1_weight = ggml_new_tensor_1d(this->ctx_weight, this->type, 100);
        this->fc1_bias   = ggml_new_tensor_1d(this->ctx_weight, this->type, 100);
        ggml_set_name(this->fc1_weight, "fc1.weight");
        ggml_set_name(this->fc1_bias,   "fc1.bias");
        this->weights.emplace("fc1.weight", this->fc1_weight);
        this->weights.emplace("fc1.bias",   this->fc1_bias);
        return *this;
    };
    // 创建计算图
    Model& build() override {
        return *this;
    };
    // 训练模型
    void train() override {
    };
    // 验证模型
    void val() override {
    };
    // 测试模型
    void test() override {
    };
    // 损失函数
    double loss(const lifuren::eval_result<int>& result) override {
        return 0.0;
    };
    // 正确数量
    double accuracy(const lifuren::eval_result<int>& result) override {
        return 0.0;
    };
    // 模型微调
    void finetune() override {
    };
    // 模型量化
    void quantization(ggml_type type) override {
    };
    // 模型预测
    int eval(int input) override {
        return 0;
    };

};

static void testSaveLoad() {
    SimpleModel save{};
    save.init().rand().build().save("D:/tmp", "simple.model");
    SimpleModel load{};
    load.load("D:/tmp", "simple.model").bind();
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testSaveLoad();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}