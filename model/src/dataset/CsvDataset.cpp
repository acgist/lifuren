#include "lifuren/Dataset.hpp"

#include <set>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/String.hpp"

// 列是不是全是数值：排除空白
static std::map<size_t, bool> digtType;
// 列的数据类型总量：排除空白
static std::map<size_t, std::map<std::string, size_t>> enumType;

static std::vector<std::vector<std::string>> loadCSV(const std::string& path);

static void loadCSV(
    const std::string&path,
    const size_t& startRow,
    const size_t& startCol,
    const int   & labelCol,
    const std::string& unknow,
    std::vector<torch::Tensor>& labels,
    std::vector<torch::Tensor>& features
);

lifuren::dataset::CsvDataset::CsvDataset(
    const std::string& path,
    const size_t& startRow,
    const size_t& startCol,
    const int   & labelCol,
    const std::string& unknow
) {
    loadCSV(path, startRow, startCol, labelCol, unknow, this->labels, this->features);
}

lifuren::dataset::CsvDataset::~CsvDataset() {
}

torch::optional<size_t> lifuren::dataset::CsvDataset::size() const {
    return this->features.size();
}

torch::data::Example<> lifuren::dataset::CsvDataset::get(size_t index) {
    return { 
        this->features[index],
        this->labels[index]
    };
}

torch::Tensor lifuren::dataset::CsvDataset::getFeature(size_t index) {
    return this->features[index];
}

void lifuren::dataset::CsvDataset::reset() {
    digtType.clear();
    enumType.clear();
}

static std::vector<std::vector<std::string>> loadCSV(const std::string& path) {
    std::ifstream stream;
    stream.open(path);
    if(!stream.is_open()) {
        stream.close();
        return {};
    }
    size_t size  = 0;
    size_t index = 0;
    size_t jndex = 0;
    std::string line;
    std::vector<std::vector<std::string>> ret;
    while(std::getline(stream, line)) {
        std::vector<std::string> data;
        data.reserve(size);
        if(line.empty()) {
            continue;
        }
        index = 0;
        jndex = 0;
        while((jndex = line.find(',', index)) != std::string::npos) {
            data.push_back(line.substr(index, jndex - index));
            index = jndex + 1;
        }
        if(line.length() > index) {
            data.push_back(line.substr(index, line.length() - index));
        }
        ret.push_back(data);
        size = data.size();
    }
    stream.close();
    return ret;
}

void loadCSV(
    const std::string&path,
    const size_t& startRow,
    const size_t& startCol,
    const int   & labelCol,
    const std::string& unknow,
    std::vector<torch::Tensor>& labels,
    std::vector<torch::Tensor>& features
) {
    const auto rows = std::move(loadCSV(path));
    if(rows.empty()) {
        return;
    }
    const size_t rowSize = rows.size();
    const size_t colSize = rows[0].size();
    if(digtType.empty() && enumType.empty()) {
        for(size_t col = startCol; col < colSize; ++col) {
            digtType[col] = true;
            for(size_t row = startRow; row < rowSize; ++row) {
                const auto& cols = rows[row];
                const auto& v    = cols[col];
                auto& typeMapping = enumType[col];
                auto iterator = typeMapping.find(v);
                if(iterator == typeMapping.end()) {
                    // 0 = 空白
                    typeMapping.emplace(v, typeMapping.size());
                }
                if(v.empty() || v == unknow) {
                } else if(lifuren::string::isNumeric(v)) {
                } else {
                    digtType[col] = false;
                }
            }
        }
    }
    size_t lSize = 0;
    size_t fSize = 0;
    labels.reserve(rowSize - startRow);
    features.reserve(rowSize - startRow);
    const size_t labelPos = labelCol < 0 ? colSize + labelCol : labelCol;
    for(size_t row = startRow; row < rowSize; ++row) {
        std::vector<float> label;
        std::vector<float> feature;
        label.reserve(lSize);
        feature.reserve(fSize);
        for(size_t col = startCol; col < colSize; ++col) {
            const auto& cols = rows[row];
            const auto& v    = cols[col];
            auto& ref = col == labelPos ? label : feature;
            if(digtType[col]) {
                if(v.empty() || v == unknow) {
                    ref.push_back(0.0F);
                } else {
                    ref.push_back(std::atof(v.c_str()));
                }
            } else {
                // 如果训练数据类型不足可能导致正确率低
                auto& typeRef = enumType[col];
                const size_t oldSize = ref.size();
                ref.resize(oldSize + typeRef.size(), 0.0F);
                auto iterator = typeRef.find(v);
                if(iterator == typeRef.end()) {
                    SPDLOG_WARN("数据枚举无效：{} - {} - {}", row, col, v);
                } else {
                    ref[oldSize + iterator->second] = 1.0F;
                }
            }
        }
        labels.push_back(std::move(torch::from_blob(label.data(), { static_cast<int>(label.size()) }, torch::kFloat32).clone()));
        features.push_back(std::move(torch::from_blob(feature.data(), { static_cast<int>(feature.size()) }, torch::kFloat32).clone()));
        lSize = label.size();
        fSize = feature.size();
    }
}
