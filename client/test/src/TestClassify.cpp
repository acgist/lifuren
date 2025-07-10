#include "lifuren/Test.hpp"

#include <memory>
#include <random>

#include "lifuren/Trainer.hpp"
#include "lifuren/Dataset.hpp"

class ClassifyImpl : public torch::nn::Module {

private:
    lifuren::config::ModelParams params;
    torch::nn::BatchNorm1d norm    { nullptr };
    torch::nn::Linear      linear_1{ nullptr };
    torch::nn::Linear      linear_2{ nullptr };

public:
    ClassifyImpl(lifuren::config::ModelParams params = {}) : params(params) {
        this->norm     = this->register_module("norm",    torch::nn::BatchNorm1d(2));
        this->linear_1 = this->register_module("linear_1", torch::nn::Linear(torch::nn::LinearOptions( 2, 16)));
        this->linear_2 = this->register_module("linear_2", torch::nn::Linear(torch::nn::LinearOptions(16,  4)));
    }
    torch::Tensor forward(torch::Tensor input) {
        auto output = this->linear_1(this->norm(input));
             output = this->linear_2(torch::relu(output));
        return output;
    }
    virtual ~ClassifyImpl() {
        this->unregister_module("norm");
        this->unregister_module("linear_1");
        this->unregister_module("linear_2");
    }

};

TORCH_MODULE(Classify);

class ClassifyTrainer : public lifuren::Trainer<torch::optim::Adam, Classify, lifuren::dataset::RndDatasetLoader> {

private:
    size_t accu_val = 0;
    size_t data_val = 0;
    torch::Tensor confusion_matrix;
    torch::nn::CrossEntropyLoss cross_entropy_loss;

public:
    ClassifyTrainer(lifuren::config::ModelParams params = {
        .lr         = 0.01F,
        .batch_size = 100,
        .epoch_size = 32
    }) : Trainer(params) {
        // 混淆矩阵 4 * 4
        this->confusion_matrix = torch::zeros({ 4, 4 }, torch::kInt).requires_grad_(false).to(torch::kCPU);
    }
    virtual ~ClassifyTrainer() {
    }

public:
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
        auto pred_index   = torch::softmax(pred, 1).argmax(1).to(torch::kCPU);
        auto batch_size   = pred_index.numel();
        auto accu = pred_index.eq(target_index).sum();
        accu_val += accu.template item<int>();
        data_val += batch_size;
        int64_t* target_index_iter = target_index.data_ptr<int64_t>();
        int64_t* pred_index_iter   = pred_index.data_ptr<int64_t>();
        for (int64_t i = 0; i < batch_size; ++i, ++target_index_iter, ++pred_index_iter) {
            confusion_matrix[*target_index_iter][*pred_index_iter].add_(1);
        }
    }
    void defineDataset() override {
        std::mt19937 rand(std::random_device{}());
        std::normal_distribution<float> w(10.0, 1.0); // 标准差越大越难拟合
        std::normal_distribution<float> b( 0.5, 0.2);
        std::vector<torch::Tensor> labels;
        std::vector<torch::Tensor> features;
        labels  .reserve(4000);
        features.reserve(4000);
        for(int index = 0; index < 4000; ++index) {
            int label = index % 4;
            float l[] = { 0, 0, 0, 0 };
            float f[] = { w(rand) * label + b(rand), w(rand) * label + b(rand) };
            l[label]  = 1.0F;
            labels  .push_back(torch::from_blob(l, { 4 }, torch::kFloat32).clone().to(lifuren::get_device()));
            features.push_back(torch::from_blob(f, { 2 }, torch::kFloat32).clone().to(lifuren::get_device()));
        }
        auto dataset = lifuren::dataset::Dataset(this->params.batch_size, labels, features).map(torch::data::transforms::Stack<>());
        this->trainDataset = torch::data::make_data_loader<LFT_RND_SAMPLER>(std::move(dataset), this->params.batch_size);
    }
    void defineOptimizer() override {
        torch::optim::AdamOptions optims;
        optims.lr (this->params.lr);
        optims.eps(0.0001);
        this->optimizer = std::make_unique<torch::optim::Adam>(this->model->parameters(), optims);
    }
    void printEvaluation(
        const char*  name,
        const size_t epoch,
        const float  loss,
        const size_t duration
    ) override {
        SPDLOG_INFO(
            "当前{}第 {} 轮，损失值为：{:.6f}，耗时：{}，正确率为：{} / {} = {:.6f}。",
            name,
            epoch,
            loss,
            duration,
            this->accu_val,
            this->data_val,
            1.0F * this->accu_val / this->data_val
        );
        std::cout << "混淆矩阵\n" << this->confusion_matrix << std::endl;
        this->accu_val = 0;
        this->data_val = 0;
        this->confusion_matrix.fill_(0);
    }
    void loss(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override {
        pred = this->model->forward(feature);
        loss = this->cross_entropy_loss->forward(pred, label);
        // 计算混淆矩阵
        this->classify_evaluate(label, pred, this->confusion_matrix, this->accu_val, this->data_val);
    }
    torch::Tensor pred(torch::Tensor feature) {
        return this->model->forward(feature);
    }

};

[[maybe_unused]] static void testTrain() {
    ClassifyTrainer classify;
    classify.define();
    classify.trainValAndTest(false, false);
    classify.print();
    classify.save();
    auto pred = torch::softmax(classify.pred(torch::tensor({ 4.0F, 4.0F }, torch::kFloat32).reshape({ 1, 2 }).to(lifuren::get_device())), 1);
    std::cout << "预测结果\n" << pred << std::endl;
    auto class_id  = pred.argmax(1);
    auto class_idx = class_id.item<int>();
    SPDLOG_INFO("预测结果：{} - {}", class_id.item().toInt(), pred[0][class_idx].item().toFloat());
}

[[maybe_unused]] static void testPred() {
    ClassifyTrainer classify;
    classify.define();
    classify.load();
    classify.print();
    std::vector<float> data = {
        0.1F,   0.2F,
        2.0F,   1.0F,
        10.0F, 11.0F,
        20.0F, 22.0F,
        30.0F, 33.0F,
        90.0F, 99.0F,
    };
    auto pred = torch::softmax(classify.pred(torch::from_blob(data.data(), { static_cast<int>(data.size()) / 2, 2 }, torch::kFloat32).to(lifuren::get_device())), 1);
    std::cout << "当前预测\n" << pred << std::endl;
    std::cout << "预测类别\n" << pred.argmax(1) << std::endl;
    std::cout << "预测类别\n" << std::get<1>(pred.max(1)) << std::endl;
    std::cout << "预测概率\n" << std::get<0>(pred.max(1)) << std::endl;
}

LFR_TEST(
    testTrain();
    // testPred();
);
