#include "lifuren/Dataset.hpp"

#include "spdlog/spdlog.h"

torch::Tensor lifuren::dataset::score::score_to_tensor(const lifuren::music::Score& score) {
    // TODO
    return {};
}

void lifuren::dataset::score::tensor_to_score(lifuren::music::Score& score, const torch::Tensor& tensor) {
    // TODO
}

lifuren::dataset::DatasetLoader lifuren::dataset::score::loadBeethovenDatasetLoader(const size_t batch_size, const std::string& path) {
    auto dataset = lifuren::dataset::Dataset(
        path,
        { ".xml" },
        [] (
            const std::string         & file,
            std::vector<torch::Tensor>& labels,
            std::vector<torch::Tensor>& features,
            const torch::DeviceType   & device
        ) {
            auto score = lifuren::music::load_xml(file);
            if(score.empty()) {
                SPDLOG_WARN("加载数据失败：{}", file);
                return;
            }
            // TODO: 指法
            auto tensor = lifuren::dataset::score::score_to_tensor(score);
            labels.push_back(tensor.clone().to(device));
            features.push_back(tensor.clone().to(device));
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<LFT_SAMPLER>(std::move(dataset), batch_size);
}
