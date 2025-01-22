#include "lifuren/Test.hpp"

#include "lifuren/RAGClient.hpp"
#include "lifuren/poetry/PoetryModel.hpp"

[[maybe_unused]] static void testSuxinTrain() {
    lifuren::poetry::SuxinModel model({
        // .epoch_count = 8,
        .check_point = true,
        // .train_path = lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "embedding.model"}).string()
        .train_path = lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "embedding_val.model"}).string()
        // .train_path = lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "embedding_train.model"}).string()
        // .val_path   = lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren", "embedding_val.model"}).string()
    });
    model.define();
    model.trainValAndTest();
    model.save(lifuren::config::CONFIG.tmp, "suxin.pt");
}

[[maybe_unused]] static void testSuxinPred() {
    lifuren::poetry::SuxinModel model;
    model.load(lifuren::config::CONFIG.tmp, "suxin.pt");
    // 预测
    const int dims = 768;
    int index = 0;
    std::vector<std::vector<float>> rule;
    torch::DeviceType device{ torch::kCPU };
    lifuren::setDevice(device);
    const auto& rhythm = lifuren::config::RHYTHM.find("蝶恋花")->second;
    SPDLOG_DEBUG("当前格律：{}", rhythm.rhythm);
    lifuren::poetry::fillRhythm(dims, rule, &rhythm);
    const auto ragClient = lifuren::RAGClient::getClient(
        "faiss",
        lifuren::file::join({lifuren::config::CONFIG.tmp, "lifuren"}).string(),
        "ollama"
    );
    const int sequenceLength = lifuren::config::CONFIG.poetry.length;
    torch::Tensor pred;
    std::vector<float> indexVector(dims, 0.0F);
    std::vector<torch::Tensor> sequence(7);
    std::fill(indexVector.begin() + index, indexVector.begin() + index + sequenceLength, 1.0F);
    sequence[0] = torch::from_blob(rule[0].data(), { dims }, torch::kFloat32).to(device).clone();
    sequence[1] = torch::from_blob(rule[1].data(), { dims }, torch::kFloat32).to(device).clone();
    sequence[2] = torch::from_blob(indexVector.data(), { dims }, torch::kFloat32).to(device).clone();
    sequence[3] = torch::from_blob(ragClient->index("昨夜").data(), { dims }, torch::kFloat32).to(device).clone();
    sequence[4] = torch::from_blob(ragClient->index("西风").data(), { dims }, torch::kFloat32).to(device).clone();
    sequence[5] = torch::from_blob(ragClient->index("凋碧树").data(), { dims }, torch::kFloat32).to(device).clone();
    sequence[6] = torch::from_blob(ragClient->index("独上").data(), { dims }, torch::kFloat32).to(device).clone();
    while(true) {
        // 不能用resize
        auto feature = torch::cat(sequence).reshape({ sequenceLength + 3, dims });
        // SPDLOG_DEBUG("f sizes: {}", feature.sizes());
        // SPDLOG_DEBUG("f sizes: {}", feature.unsqueeze(0).permute({ 1, 0, 2 }).sizes());
        pred = model.pred(feature.unsqueeze(0).permute({ 1, 0, 2 }));
        // SPDLOG_DEBUG("p sizes: {}", pred.sizes());
        std::vector<float> prompt(pred.data_ptr<float>(), pred.data_ptr<float>() + pred.numel());
        std::vector<std::string> words = ragClient->search(prompt, 4);
        SPDLOG_DEBUG("{} - {} - {} - {}", words[0], words[1], words[2], words[3]);
        if(pred.count_nonzero().item<int>() == 0) {
            break;
        }
        // SPDLOG_DEBUG("p zero size: {}", pred.count_nonzero().item<int>());
        ++index;
        std::fill(indexVector.begin(), indexVector.end(), 0.0F);
        std::fill(indexVector.begin() + index, indexVector.begin() + index + sequenceLength, 1.0F);
        sequence[2] = torch::from_blob(indexVector.data(), { dims }, torch::kFloat32).to(device).clone();
        sequence[3] = sequence[4];
        sequence[4] = sequence[5];
        sequence[5] = sequence[6];
        sequence[6] = pred;
    }
}

LFR_TEST(
    // testSuxinTrain();
    testSuxinPred();
);
