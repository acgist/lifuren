#include "lifuren/PoetryDatasets.hpp"

#include <functional>

extern bool lifuren::datasets::poetry::read(std::ifstream& stream, std::vector<std::vector<float>>& vector) {
    short size{ 0 };
    if(stream >> size && size != lifuren::datasets::poetry::END_OF_POETRY) {
        std::vector<float> v;
        v.reserve(size);
        float x{ 0.0F };
        for(int i = 0; i < size; ++i) {
            if(stream >> x) {
                v.push_back(x);
            }
        }
        vector.push_back(std::move(v));
    }
    return size == lifuren::datasets::poetry::END_OF_DATASETS;
}

extern void lifuren::datasets::poetry::write(std::ofstream& stream, std::vector<std::vector<float>>& vector) {
    std::for_each(vector.begin(), vector.end(), [&stream](const auto& v) {
        stream << static_cast<short>(v.size());
        std::for_each(v.begin(), v.end(), [&stream](const auto& x) {
            stream << x;
        });
    });
    stream << lifuren::datasets::poetry::END_OF_POETRY;
}
