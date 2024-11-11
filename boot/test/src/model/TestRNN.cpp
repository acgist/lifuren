#include "lifuren/Test.hpp"

#include <random>

#include "lifuren/Model.hpp"
#include "lifuren/Dataset.hpp"

[[maybe_unused]] static void testGRU() {
    // https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
    torch::nn::GRUOptions options(10, 20); // input_size hidden_size
    // options.num_layers(2);
    torch::nn::GRU gru(options);
    // auto i0 = torch::randn({ 5, 3, 10 }); // L N input_size
    // auto h0 = torch::randn({ 1, 3, 20 }); // D[1|2] * num_layers N hidden_size
    // auto [o1, h1] = gru->forward(i0, h0);
    // SPDLOG_DEBUG("o sizes:\n{}", o1.sizes());
    // SPDLOG_DEBUG("h sizes:\n{}", h1.sizes());
    auto i0 = torch::randn({ 5, 1, 10 });
    auto h0 = torch::randn({ 1, 1, 20 });
    auto [o1, h1] = gru->forward(i0, h0);
    SPDLOG_DEBUG("o1 sizes:\n{}", o1.sizes());
    SPDLOG_DEBUG("h1 sizes:\n{}", h1.sizes());
    auto i1 = torch::randn({ 5, 1, 10 });
    auto [o2, h2] = gru->forward(i1, h1);
    SPDLOG_DEBUG("o2 sizes:\n{}", o2.sizes());
    SPDLOG_DEBUG("h2 sizes:\n{}", h2.sizes());
}

[[maybe_unused]] static void testLSTM() {
    // https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    torch::nn::LSTMOptions options(2, 4);
    torch::nn::LSTM lstm(options);
    auto input = torch::randn({ 5, 3, 2 });
    auto h0    = torch::randn({ 1, 3, 4 });
    auto c0    = torch::randn({ 1, 3, 4 });
    auto [output, v] = lstm->forward(input, std::make_tuple<>(h0, c0));
    auto [hn, cn] = v;
    // auto [output, [hn, cn]] = lstm->forward(input, std::make_tuple<>(h0, c0));
    SPDLOG_DEBUG("o sizes:\n{}", output.sizes());
    SPDLOG_DEBUG("h sizes:\n{}", hn.sizes());
    SPDLOG_DEBUG("c sizes:\n{}", cn.sizes());
}

class RNNModuleImpl : public torch::nn::Module {

private:
    torch::nn::GRU    gru   { nullptr };
    torch::nn::Linear linear{ nullptr };
    // torch::Tensor  hidden;

public:
    RNNModuleImpl() {
        // input // L N input_size = 句子长度 批量数量 词语维度
        // this->hidden = torch::randn({1, 10, 1}); // D[1|2] * num_layers N hidden_size
        torch::nn::GRU gru(torch::nn::GRUOptions(1, 1)); // input_size hidden_size
        this->gru = register_module("gru", gru);
        torch::nn::Linear linear(torch::nn::LinearOptions(4, 1));
        this->linear = register_module("linear", linear);
    }
    virtual ~RNNModuleImpl() {
        unregister_module("gru");
        unregister_module("linear");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        // SPDLOG_DEBUG("{}", input.sizes());
        // SPDLOG_DEBUG("{}", this->hidden.sizes());
        auto [output, hidden] = this->gru->forward(input);
        // auto [output, hidden] = this->gru->forward(input, this->hidden);
        // SPDLOG_DEBUG("{}", output.sizes());
        // SPDLOG_DEBUG("{}", hidden.sizes());
        // SPDLOG_DEBUG("{}", output.squeeze().sizes());
        return this->linear->forward(output.squeeze().t());
    }

};

TORCH_MODULE(RNNModule);

class RNNModel : public lifuren::Model<
    lifuren::dataset::RawDatasetLoader,
    torch::Tensor,
    torch::Tensor,
    torch::nn::MSELoss,
    // torch::nn::CrossEntropyLoss,
    RNNModule,
    torch::optim::Adam
> {

public:
    RNNModel(lifuren::ModelParams params = {}) : Model(params) {
    }
    virtual ~RNNModel() {
    }

public:
    bool defineDataset() override {
        std::random_device device;
        std::mt19937 rand(device());
        std::normal_distribution<float> nd(0.5, 0.2);
        std::vector<torch::Tensor> labels;
        std::vector<torch::Tensor> features;
        const int count = 200;
        labels.reserve(count);
        features.reserve(count);
        for(int index = 0; index < count; ++index) {
            // const float a0 = nd(rand);
            const float a0 = nd(rand) * 10;
            const float a1 = a0 + 0 + nd(rand);
            const float a2 = a1 + 1 + nd(rand);
            const float a3 = a2 + 2 + nd(rand);
            const float a4 = a3 + 3 + nd(rand);
            const float a5 = a4 + 4 + nd(rand);
            float data[] { a1, a2, a3, a4 };
            features.push_back(torch::from_blob(data, { 4, 1 }, torch::kFloat32).clone());
            labels.push_back(torch::tensor({ a5 }));
        }
        this->trainDataset = std::move(lifuren::dataset::loadRawDataset(this->params.batch_size, labels, features));
        return true;
    }
    void logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override {
        // SPDLOG_DEBUG("feature = {}", feature.sizes());
        feature = feature.permute({1, 0, 2});
        // SPDLOG_DEBUG("feature = {}", feature.sizes());
        pred = std::move(this->model->forward(feature));
        // SPDLOG_DEBUG("pred  = {}", pred.sizes());
        // SPDLOG_DEBUG("label = {}", label.sizes());
        loss = std::move(this->loss->forward(pred, label));
    }
    torch::Tensor pred(torch::Tensor i) override {
        i = i.unsqueeze(0).unsqueeze(0).permute({ 2, 1, 0 });
        SPDLOG_DEBUG("i = {}", i.sizes());
        return this->model->forward(i);
    }

};

[[maybe_unused]] static void testRNNModel() {
    RNNModel save({
        .lr          = 0.1F,
        .batch_size  = 10,
        .epoch_count = 32
    });
    save.define();
    // save.print();
    save.trainValAndTest();
    auto pred1 = save.pred(torch::tensor({ 1.0F, 2.0F, 4.0F, 7.0F })); // 7 + 4
    SPDLOG_DEBUG("pred = \n{}", pred1.sizes());
    SPDLOG_DEBUG("pred = \n{}", pred1.squeeze());
    auto pred2 = save.pred(torch::tensor({ 2.0F, 3.0F, 5.0F, 8.0F })); // 8 + 4
    SPDLOG_DEBUG("pred = \n{}", pred2.sizes());
    SPDLOG_DEBUG("pred = \n{}", pred2.squeeze());
}

LFR_TEST(
    // testGRU();
    // testLSTM();
    testRNNModel();
);
