#include "lifuren/Dataset.hpp"

lifuren::dataset::ShardingDataset::ShardingDataset(
    std::shared_ptr<Dataset> source,
    std::map<size_t, size_t> indexMapping
) : lifuren::dataset::Dataset(indexMapping.size() * source->getBatchSize(), source->getBatchSize()),
    source(source),
    indexMapping(std::move(indexMapping))
{
}

lifuren::dataset::ShardingDataset::~ShardingDataset() {
}

std::tuple<lifuren::dataset::ShardingDataset, lifuren::dataset::ShardingDataset, lifuren::dataset::ShardingDataset> lifuren::dataset::ShardingDataset::make(
    std::shared_ptr<Dataset> source,
    size_t valIndex,
    size_t valCount,
    size_t testIndex,
    size_t testCount
) {
    std::map<size_t, size_t> trainMapping;
    std::map<size_t, size_t> valMapping;
    std::map<size_t, size_t> testMapping;
    const size_t count = source->getBatchCount();
    size_t trainPos = 0LL;
    size_t valPos   = 0LL;
    size_t testPos  = 0LL;
    for(size_t index = 0LL; index < count; ++index) {
        if(valCount > 0LL && valIndex <= index && index < valIndex + valCount) {
            valMapping.emplace(valPos++, index);
        } else if(testCount > 0LL && testIndex <= index && index < testIndex + testCount) {
            testMapping.emplace(testPos++, index);
        } else {
            trainMapping.emplace(trainPos++, index);
        }
    }
    return std::make_tuple(
        ShardingDataset(source, trainMapping),
        ShardingDataset(source, valMapping),
        ShardingDataset(source, testMapping)
    );
}
