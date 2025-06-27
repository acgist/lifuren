#include "lifuren/Dataset.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Config.hpp"

std::vector<std::string> lifuren::dataset::allDataset(const std::string& path) {
    std::vector<std::string> ret;
    ret.reserve(3);
    const auto train_path = lifuren::file::join({ path, lifuren::config::DATASET_TRAIN });
    const auto val_path   = lifuren::file::join({ path, lifuren::config::DATASET_VAL   });
    const auto test_path  = lifuren::file::join({ path, lifuren::config::DATASET_TEST  });
    if(std::filesystem::exists(train_path)) {
        ret.push_back(train_path.string());
    } else {
        SPDLOG_DEBUG("无效的训练数据集：{}", train_path.string());
    }
    if(std::filesystem::exists(val_path)) {
        ret.push_back(val_path.string());
    } else {
        SPDLOG_DEBUG("无效的验证训练集：{}", val_path.string());
    }
    if(std::filesystem::exists(test_path)) {
        ret.push_back(test_path.string());
    } else {
        SPDLOG_DEBUG("无效的测试训练集：{}", test_path.string());
    }
    return ret;
}

lifuren::dataset::Dataset::Dataset(
    bool time_seq,
    size_t batch_size,
    std::vector<torch::Tensor>& labels,
    std::vector<torch::Tensor>& features
) : time_seq(time_seq), batch_size(batch_size), device(lifuren::getDevice()), labels(std::move(labels)), features(std::move(features)) {
}

lifuren::dataset::Dataset::Dataset(
    bool time_seq,
    size_t batch_size,
    const std::string& path,
    const std::vector<std::string>& suffix,
    const std::function<void(const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform
) : time_seq(time_seq), batch_size(batch_size), device(lifuren::getDevice()) {
    if(!lifuren::file::exists(path) || !lifuren::file::is_directory(path)) {
        SPDLOG_WARN("数据集无效：{}", path);
        return;
    }
    std::vector<std::string> files;
    lifuren::file::list_file(files, path, suffix);
    for(const auto& file : files) {
        SPDLOG_DEBUG("加载文件：{}", file);
        transform(file, this->labels, this->features, this->device);
    }
}

lifuren::dataset::Dataset::~Dataset() {
}

torch::optional<size_t> lifuren::dataset::Dataset::size() const {
    return this->labels.size();
}

torch::data::Example<> lifuren::dataset::Dataset::get(size_t index) {
    if(this->time_seq) {
        size_t row_size = this->labels.size() / this->batch_size;
        size_t row = index / this->batch_size;
        size_t col = index % this->batch_size;
        return {
            this->features[col * row_size + row],
            this->labels  [col * row_size + row]
        };
    } else {
        return {
            this->features[index],
            this->labels  [index]
        };
    }
}
