#include "lifuren/Dataset.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Config.hpp"

std::vector<std::string> lifuren::dataset::allDataset(const std::string& path) {
    std::vector<std::string> ret(3);
    const auto train_path = lifuren::file::join({ path, lifuren::config::DATASET_TRAIN });
    const auto val_path   = lifuren::file::join({ path, lifuren::config::DATASET_VAL   });
    const auto test_path  = lifuren::file::join({ path, lifuren::config::DATASET_TEST  });
    if(std::filesystem::exists(train_path)) {
        ret.push_back(train_path.string());
    } else {
        SPDLOG_DEBUG("没有训练数据集：{}", train_path.string());
    }
    if(std::filesystem::exists(val_path)) {
        ret.push_back(val_path.string());
    } else {
        SPDLOG_DEBUG("没有验证训练集：{}", val_path.string());
    }
    if(std::filesystem::exists(test_path)) {
        ret.push_back(test_path.string());
    } else {
        SPDLOG_DEBUG("没有测试训练集：{}", test_path.string());
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
            stream.close();
            SPDLOG_WARN("数据集文件打开失败：{}", model_path.string());
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
    const std::function<void(const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform
) : device(lifuren::getDevice()) {
    if(!lifuren::file::exists(path) || !lifuren::file::is_directory(path)) {
        SPDLOG_WARN("目录无效：{}", path);
        return;
    }
    SPDLOG_DEBUG("计算设备：{}", torch::DeviceTypeName(this->device));
    std::vector<std::string> files;
    lifuren::file::listFile(files, path, suffix);
    for(const auto& file : files) {
        SPDLOG_DEBUG("加载文件：{}", file);
        transform(file, this->labels, this->features, this->device);
    }
}

lifuren::dataset::Dataset::Dataset(
    const std::string             & path,
    const std::string             & l_suffix,
    const std::vector<std::string>& f_suffix,
    const std::function<void(const std::string&, const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform
) : device(lifuren::getDevice()) {
    if(!lifuren::file::exists(path) || !lifuren::file::is_directory(path)) {
        SPDLOG_WARN("目录无效：{}", path);
        return;
    }
    SPDLOG_DEBUG("计算设备：{}", torch::DeviceTypeName(this->device));
    std::vector<std::string> files;
    lifuren::file::listFile(files, path, f_suffix);
    for(const auto& file : files) {
        auto label_file = lifuren::file::modify_filename(file, l_suffix);
        if(!lifuren::file::exists(label_file)) {
            SPDLOG_WARN("文件没有标签：{} - {}", label_file, file);
            continue;
        }
        SPDLOG_DEBUG("加载文件：{} - {}", label_file, file);
        transform(label_file, file, this->labels, this->features, this->device);
    }
}

lifuren::dataset::Dataset::Dataset(
    const std::string                 & path,
    const std::vector<std::string>    & suffix,
    const std::map<std::string, float>& classify,
    const std::function<torch::Tensor(const std::string&, const torch::DeviceType&)> transform
) : device(lifuren::getDevice()) {
    if(!lifuren::file::exists(path) || !lifuren::file::is_directory(path)) {
        SPDLOG_WARN("目录无效：{}", path);
        return;
    }
    SPDLOG_DEBUG("计算设备：{}", torch::DeviceTypeName(this->device));
    auto iterator = std::filesystem::directory_iterator(std::filesystem::path(path));
    for(const auto& entry : iterator) {
        const auto entry_path = entry.path();
        if(entry.is_directory() && entry_path.string() != lifuren::config::LIFUREN_HIDDEN_FILE) {
            SPDLOG_DEBUG("加载文件：{}", entry_path.string());
            std::vector<std::string> files;
            lifuren::file::listFile(files, entry_path.string(), suffix);
            for(const auto& file : files) {
                SPDLOG_DEBUG("加载文件：{}", file);
                this->features.push_back(std::move(transform(file, this->device)));
            }
            this->labels.resize(this->features.size(), torch::full({ 1 }, classify.at(entry_path.filename().string()), torch::kFloat32).to(this->device));
        } else {
            SPDLOG_DEBUG("忽略文件：{}", entry_path.string());
        }
    }
}

lifuren::dataset::Dataset::~Dataset() {
}

torch::optional<size_t> lifuren::dataset::Dataset::size() const {
    return this->features.size() | this->labels.size();
}

torch::data::Example<> lifuren::dataset::Dataset::get(size_t index) {
    return {
        this->features[index],
        this->labels  [index]
    };
}
