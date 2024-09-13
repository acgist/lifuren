#include "lifuren/Datasets.hpp"

#include <cstring>

lifuren::datasets::RawDataset::RawDataset(
    size_t count,
    size_t batchSize,
    float* features,
    size_t feature_size,
    float* labels,
    size_t label_size
) : lifuren::datasets::Dataset(count, batchSize), features(features), feature_size(feature_size), labels(labels), label_size(label_size) {
}

lifuren::datasets::RawDataset::~RawDataset() {
}

size_t lifuren::datasets::RawDataset::batchGet(size_t batch, float* features, float* labels) const {
    const size_t remaining = this->count - this->batchSize * batch;
    if(remaining <= 0) {
        return 0;
    }
    if(remaining >= this->batchSize) {
        std::memcpy(features, this->features + batch * this->batchSize * feature_size, sizeof(float) * this->batchSize * feature_size);
        std::memcpy(labels,   this->labels   + batch * this->batchSize * label_size,   sizeof(float) * this->batchSize * label_size);
        return this->batchSize;
    } else {
        std::memcpy(features, this->features + batch * this->batchSize * feature_size, sizeof(float) * remaining * feature_size);
        std::memcpy(labels,   this->labels   + batch * this->batchSize * label_size,   sizeof(float) * remaining * label_size);
        return remaining;
    }
}
