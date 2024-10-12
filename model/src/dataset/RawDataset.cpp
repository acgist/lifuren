#include "lifuren/Dataset.hpp"

#include <memory>

lifuren::dataset::RawDataset::RawDataset(
    size_t count,
    size_t batchSize,
    float* features,
    size_t feature_size,
    float* labels,
    size_t label_size
) : lifuren::dataset::Dataset(count, batchSize),
    features(features),
    feature_size(feature_size),
    labels(labels),
    label_size(label_size)
{
}

lifuren::dataset::RawDataset::~RawDataset() {
}

size_t lifuren::dataset::RawDataset::batchGet(size_t batch, float* features, float* labels) const {
    if(features == nullptr && labels == nullptr) {
        return 0LL;
    }
    const size_t remaining = this->count - this->batchSize * batch;
    if(remaining <= 0LL) {
        return 0LL;
    }
    if(remaining >= this->batchSize) {
        std::copy_n(this->features + batch * this->batchSize * feature_size, this->batchSize * feature_size, features);
        // std::memcpy(features, this->features + batch * this->batchSize * feature_size, sizeof(float) * this->batchSize * feature_size);
        if(labels != nullptr && this->labels != nullptr) {
            std::copy_n(this->labels + batch * this->batchSize * label_size, this->batchSize * label_size, labels);
            // std::memcpy(labels, this->labels + batch * this->batchSize * label_size, sizeof(float) * this->batchSize * label_size);
        }
        return this->batchSize;
    } else {
        std::copy_n(this->features + batch * this->batchSize * feature_size, remaining * feature_size, features);
        // std::memcpy(features, this->features + batch * this->batchSize * feature_size, sizeof(float) * remaining * feature_size);
        if(labels != nullptr && this->labels != nullptr) {
            std::copy_n(this->labels + batch * this->batchSize * label_size, remaining * label_size, labels);
            // std::memcpy(labels, this->labels + batch * this->batchSize * label_size, sizeof(float) * remaining * label_size);
        }
        return remaining;
    }
}
