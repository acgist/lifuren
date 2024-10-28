#include "lifuren/Dataset.hpp"

#include "lifuren/Torch.hpp"

lifuren::dataset::RawDataset::RawDataset(
    std::vector<float>& labels,
    std::vector<std::vector<float>>& features
) : labels(std::move(labels)), features(std::move(features)) {
    lifuren::setDevice(this->device);
}

lifuren::dataset::RawDataset::~RawDataset() {
}

torch::optional<size_t> lifuren::dataset::RawDataset::size() const {
    return this->features.size();
}

torch::data::Example<> lifuren::dataset::RawDataset::get(size_t index) {
    auto& feature = this->features[index];
    auto& label   = this->labels[index];
    return {
        torch::from_blob(feature.data(), { static_cast<int>(feature.size()) }, torch::kFloat32).to(this->device),
        torch::from_blob(&label,         { 1                                }, torch::kFloat32).to(this->device)
    };
}
