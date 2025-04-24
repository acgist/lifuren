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

bool lifuren::dataset::allDatasetPreprocess(
    const std::string& path,
    const std::string& model_name,
    std::function<bool(const std::string&, const std::string&, std::ofstream&, lifuren::thread::ThreadPool&)> preprocess,
    bool model_base
) {
    const std::vector<std::string> datasets = lifuren::dataset::allDataset(path);
    if(datasets.empty()) {
        SPDLOG_WARN("没有数据集：{}", path);
        return false;
    }
    lifuren::thread::ThreadPool pool;
    return std::all_of(datasets.begin(), datasets.end(), [&path, &pool, &model_name, model_base, &preprocess](const auto& dataset) {
        std::ofstream stream;
        std::filesystem::path model_path;
        if(model_base) {
            model_path = lifuren::file::join({ path,    lifuren::config::LIFUREN_HIDDEN_FILE, model_name });
        } else {
            model_path = lifuren::file::join({ dataset, lifuren::config::LIFUREN_HIDDEN_FILE, model_name });
        }
        lifuren::file::createParent(model_path);
        stream.open(model_path, std::ios_base::binary);
        if(!stream.is_open()) {
            SPDLOG_WARN("打开数据集文件失败：{}", model_path.string());
            return false;
        }
        const bool ret = preprocess(path, dataset, stream, pool);
        pool.awaitTermination();
        stream.close();
        return ret;
    });
}

lifuren::dataset::Dataset::Dataset(
    std::vector<torch::Tensor>& labels,
    std::vector<torch::Tensor>& features
) : device(lifuren::getDevice()), labels(std::move(labels)), features(std::move(features)) {
}

lifuren::dataset::Dataset::Dataset(
    const std::string             & path,
    const std::vector<std::string>& suffix,
    const std::function<void(const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform,
    const std::function<void(std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> complete
) : device(lifuren::getDevice()) {
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
    if(complete) {
        complete(this->labels, this->features, this->device);
    }
}

lifuren::dataset::Dataset::~Dataset() {
}

torch::optional<size_t> lifuren::dataset::Dataset::size() const {
    return this->labels.size();
    // return this->features.size();
}

torch::data::Example<> lifuren::dataset::Dataset::get(size_t index) {
    return {
        this->features[index],
        this->labels  [index]
    };
}
