// #include "lifuren/Test.hpp"

// #include <random>
// #include <memory>

// #include "torch/torch.h"

// #include "lifuren/Model.hpp"
// #include "lifuren/Layer.hpp"
// #include "lifuren/Tensor.hpp"
// #include "lifuren/Dataset.hpp"

// class GenderModel : public lifuren::Model {

// public:
//     std::unique_ptr<lifuren::layer::Conv2d> c1{ nullptr };
//     std::unique_ptr<lifuren::layer::Conv2d> c2{ nullptr };
//     std::unique_ptr<lifuren::layer::Conv2d> c3{ nullptr };
//     std::unique_ptr<lifuren::layer::Conv2d> c4{ nullptr };
//     std::unique_ptr<lifuren::layer::Linear> l1{ nullptr };
//     std::unique_ptr<lifuren::layer::Linear> l2{ nullptr };
//     std::unique_ptr<lifuren::layer::Linear> l3{ nullptr };

// public:
//     GenderModel(lifuren::Model::ModelParams params = {}) : Model(params) {
//     }
//     ~GenderModel() {
//     }

// public:
//     Model& defineWeight() override {
//         this->c1 = lifuren::layer::conv2d(3, 8,   3, this->ctx_weight, this->ctx_compute, "c1");
//         // this->c2 = lifuren::layer::conv2d(8, 16,  3, this->ctx_weight, this->ctx_compute, "c2");
//         // this->c3 = lifuren::layer::conv2d(16, 16, 3, this->ctx_weight, this->ctx_compute, "c3");
//         // this->c4 = lifuren::layer::conv2d(16, 16, 3, this->ctx_weight, this->ctx_compute, "c4");
//         // this->l1 = lifuren::layer::linear(16 * 116 * 116, 1024, this->ctx_weight, this->ctx_compute, "l1");
//         // this->l2 = lifuren::layer::linear(1024,           256,  this->ctx_weight, this->ctx_compute, "l2");
//         // this->l3 = lifuren::layer::linear(256,            2,    this->ctx_weight, this->ctx_compute, "l3");
//         this->l3 = lifuren::layer::linear(8 * 126 * 126,            2,    this->ctx_weight, this->ctx_compute, "l3");
//         this->c1->defineWeight();
//         // this->c2->defineWeight();
//         // this->c3->defineWeight();
//         // this->c4->defineWeight();
//         // this->l1->defineWeight();
//         // this->l2->defineWeight();
//         this->l3->defineWeight();
//         return *this;
//     }
//     Model& bindWeight(const std::map<std::string, ggml_tensor*> weights) override {
//         // TODO: 不测模型保存忽略
//         return *this;
//     }
//     ggml_tensor* buildFeatures() override {
//         return ggml_new_tensor_4d(this->ctx_compute, GGML_TYPE_F32, 128, 128, 3, this->params.batch_size);
//     };
//     ggml_tensor* buildLabels() override {
//         return ggml_new_tensor_2d(this->ctx_compute, GGML_TYPE_F32, 2, this->params.batch_size);
//     };
//     ggml_tensor* buildLoss() override {
//         return ggml_cross_entropy_loss(this->ctx_compute, this->logits, this->labels);
//     };
//     ggml_tensor* buildLogits() override {
//         ggml_tensor* ret = this->c1->forward(this->features);
//         // ret = this->c2->forward(ret);
//         // ret = lifuren::function::maxPool2d(3, ret, this->ctx_compute);
//         // ret = this->c3->forward(ret);
//         // ret = this->c4->forward(ret);
//         // ret = lifuren::function::avgPool2d(3, ret, this->ctx_compute);
//         // ret = ggml_reshape_2d(this->ctx_compute, ggml_cont(this->ctx_compute, ggml_permute(this->ctx_compute, ret, 1, 2, 0, 3)), 16 * 116 * 116, this->params.batch_size);
//         ret = ggml_reshape_2d(this->ctx_compute, ggml_cont(this->ctx_compute, ggml_permute(this->ctx_compute, ret, 1, 2, 0, 3)), 8 * 126 * 126, this->params.batch_size);
//         // // ret = ggml_relu(this->ctx_compute, this->l1->forward(ret));
//         // // ret = ggml_relu(this->ctx_compute, this->l2->forward(ret));
//         // ret = ggml_relu_inplace(this->ctx_compute, this->l1->forward(ret));
//         // ret = ggml_relu_inplace(this->ctx_compute, this->l2->forward(ret));
//         // ret = ggml_relu_inplace(this->ctx_compute, this->l3->forward(ret));
//         ret = this->l3->forward(ret);
//         // ret = ggml_soft_max_inplace(this->ctx_compute, ret);
//         return ret;
//     };

// };

// [[maybe_unused]] static void testGender() {
//     std::random_device device;
//     std::mt19937 rand(device());
//     std::normal_distribution<> weight(10, 2);
//     std::normal_distribution<> bias  (0 , 2);
//     float features[210];
//     float labels  [210];
//     for(int index = 0; index < 210; ++index) {
//         features[index] = weight(rand);
//         labels  [index] = features[index] * 15.4 + 4 + bias(rand);
//     }
//     lifuren::dataset::RawDataset* dataset = new lifuren::dataset::RawDataset(
//         210,
//         10,
//         features,
//         1,
//         labels,
//         1
//     );
//     lifuren::Model::OptimizerParams optParams {
//         .n_iter = 20
//     };
//     lifuren::Model::ModelParams params {
//         .batch_size      = 10,
//         .epoch_count     = 256,
//         // .epoch_count  = 1024,
//         .classify        = true,
//         .size_weight     = 128LL * 1024 * 1024,
//         .size_compute    = 512LL * 1024 * 1024,
//         // .size_weight     = 1024LL * 1024 * 1024,
//         // .size_compute    = 4096LL * 1024 * 1024,
//         .optimizerParams = optParams
//     };
//     GenderModel save{params};
//     save.trainDataset.reset(dataset);
//     save.define();
//     // save.save(lifuren::config::CONFIG.tmp);
//     // save.print();
//     save.trainValAndTest(false, false);
//     float data[] { 3.2 };
//     float target[1];
//     // w * 15.4 + 4 + rand
//     float* pred = save.eval(data, target, 1);
//     SPDLOG_DEBUG("当前预测：{}", *pred);
// }

// LFR_TEST(
//     testGender();
// );
