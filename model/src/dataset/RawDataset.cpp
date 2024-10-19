#include "lifuren/Dataset.hpp"

#include <memory>

lifuren::dataset::RawDataset::RawDataset(
    const std::vector<float>& labels,
    const std::vector<std::vector<float>>& features
) : labels(labels),
    features(features)
{
}

lifuren::dataset::RawDataset::~RawDataset() {
}

torch::optional<size_t> lifuren::dataset::RawDataset::size() const {
    return this->features.size();
}

torch::data::Example<> lifuren::dataset::RawDataset::get(size_t index) {
    auto& feature = this->features.at(index);
    auto& label   = this->labels.at(index);
    return {
        torch::from_blob(feature.data(), { static_cast<int>(feature.size()) }, torch::kFloat32),
        torch::from_blob(&label,         { 1                                }, torch::kFloat32)
    };
}
