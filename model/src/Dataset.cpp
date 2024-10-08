#include "lifuren/Datasets.hpp"

lifuren::datasets::Dataset::Dataset(size_t batchSize) : batchSize(batchSize) {
}

lifuren::datasets::Dataset::Dataset(size_t count, size_t batchSize) : count(count), batchSize(batchSize) {
}

lifuren::datasets::Dataset::~Dataset() {
}

const size_t& lifuren::datasets::Dataset::getCount() const {
    return this->count;
}

const size_t& lifuren::datasets::Dataset::getBatchSize() const {
    return this->batchSize;
}

size_t lifuren::datasets::Dataset::getBatchCount() const {
    if(this->count % this->batchSize == 0) {
        return this->count / this->batchSize;
    } else {
        return this->count / this->batchSize + 1;
    }
}
