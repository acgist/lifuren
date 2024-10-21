#include "lifuren/PoetryDataset.hpp"

#include <functional>

#include "spdlog/spdlog.h"

bool lifuren::dataset::poetry::read(std::ifstream& stream, std::vector<std::vector<float>>& vector) {
    short size{ 0 };
    if(stream.read(reinterpret_cast<char*>(&size), sizeof(size)) && size > 0) {
        std::vector<float> v;
        v.resize(size);
        stream.read(reinterpret_cast<char*>(v.data()), sizeof(float) * size);
        vector.push_back(std::move(v));
    }
    return size == lifuren::dataset::poetry::END_OF_DATASET;
}

void lifuren::dataset::poetry::write(std::ofstream& stream, const std::vector<std::vector<float>>& vector) {
    std::for_each(vector.begin(), vector.end(), [&stream](const std::vector<float>& v) {
        const short size = static_cast<short>(v.size());
        stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
        stream.write(reinterpret_cast<const char*>(v.data()), sizeof(float) * size);
    });
    lifuren::dataset::poetry::writeEnd(stream, lifuren::dataset::poetry::END_OF_POETRY);
}

bool lifuren::dataset::poetry::fillRhythm(const int& dims, std::vector<std::vector<float>>& vector, const lifuren::config::Rhythm* rhythm) {
    if(rhythm == nullptr) {
        return false;
    }
    // TODO: 平仄维度
    if(dims < rhythm->fontSize) {
        SPDLOG_WARN("诗词长度超过向量维度：{} - {} - {}", rhythm->rhythm, rhythm->fontSize, dims);
        return false;
    }
    std::vector<float> segmentRule;
    segmentRule.resize(dims, 0.0F);
    std::for_each(rhythm->segmentRule.begin(), rhythm->segmentRule.end(), [pos = 0, &segmentRule](const auto& index) mutable {
        segmentRule[pos += index] = 1.0F;
    });
    std::vector<float> participleRule;
    participleRule.resize(dims, 0.0F);
    std::for_each(rhythm->participleRule.begin(), rhythm->participleRule.end(), [pos = 0, &participleRule](const auto& index) mutable {
        participleRule[pos += index] = 1.0F;
    });
    vector.push_back(segmentRule);
    vector.push_back(participleRule);
    return true;
}
