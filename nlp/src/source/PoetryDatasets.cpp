#include "lifuren/PoetryDatasets.hpp"

#include <functional>

bool lifuren::datasets::poetry::read(std::ifstream& stream, std::vector<std::vector<float>>& vector) {
    short size{ 0 };
    if(stream.read(reinterpret_cast<char*>(&size), sizeof(size)) && size > 0) {
        std::vector<float> v;
        v.resize(size);
        stream.read(reinterpret_cast<char*>(v.data()), sizeof(float) * size);
        vector.push_back(std::move(v));
    }
    return size == lifuren::datasets::poetry::END_OF_DATASETS;
}

void lifuren::datasets::poetry::write(std::ofstream& stream, std::vector<std::vector<float>>& vector) {
    std::for_each(vector.begin(), vector.end(), [&stream](const std::vector<float>& v) {
        const short size = static_cast<short>(v.size());
        stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
        stream.write(reinterpret_cast<const char*>(v.data()), sizeof(float) * size);
    });
    stream.write(reinterpret_cast<const char*>(&lifuren::datasets::poetry::END_OF_POETRY), sizeof(lifuren::datasets::poetry::END_OF_POETRY));
}
