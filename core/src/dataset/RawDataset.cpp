#include "lifuren/Dataset.hpp"

#include "lifuren/Torch.hpp"

lifuren::dataset::RawDataset::RawDataset(
    std::vector<torch::Tensor>& labels,
    std::vector<torch::Tensor>& features
) : labels(std::move(labels)), features(std::move(features)) {
    lifuren::setDevice(this->device);
}

lifuren::dataset::RawDataset::~RawDataset() {
}

torch::optional<size_t> lifuren::dataset::RawDataset::size() const {
    return this->features.size();
}

torch::data::Example<> lifuren::dataset::RawDataset::get(size_t index) {
    return {
        this->features[index],
        this->labels[index]
    };
}
