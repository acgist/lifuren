#include "lifuren/Dataset.hpp"

lifuren::dataset::Dataset::Dataset(size_t batchSize) : batchSize(batchSize) {
}

lifuren::dataset::Dataset::Dataset(size_t count, size_t batchSize) : count(count), batchSize(batchSize) {
}

lifuren::dataset::Dataset::~Dataset() {
}

const size_t& lifuren::dataset::Dataset::getCount() const {
    return this->count;
}

const size_t& lifuren::dataset::Dataset::getBatchSize() const {
    return this->batchSize;
}

size_t lifuren::dataset::Dataset::getBatchCount() const {
    if(this->count % this->batchSize == 0) {
        return this->count / this->batchSize;
    } else {
        return this->count / this->batchSize + 1;
    }
}
