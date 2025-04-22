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

lifuren::dataset::Dataset::Dataset(
    const std::string             & path,
    const std::string             & l_suffix,
    const std::vector<std::string>& f_suffix,
    const std::function<void(const std::string&, const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform,
    const std::function<void(std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> complete
) : device(lifuren::getDevice()) {
    if(!lifuren::file::exists(path) || !lifuren::file::is_directory(path)) {
        SPDLOG_WARN("数据集无效：{}", path);
        return;
    }
    std::vector<std::string> files;
    lifuren::file::list_file(files, path, f_suffix);
    for(const auto& f_file : files) {
        auto l_file = lifuren::file::modify_filename(f_file, l_suffix);
        if(!lifuren::file::exists(l_file)) {
            SPDLOG_WARN("特征没有标签：{} - {}", l_file, f_file);
            continue;
        }
        SPDLOG_DEBUG("加载文件：{} - {}", l_file, f_file);
        transform(l_file, f_file, this->labels, this->features, this->device);
    }
    if(complete) {
        complete(this->labels, this->features, this->device);
    }
}

lifuren::dataset::Dataset::Dataset(
    const std::string                 & path,
    const std::vector<std::string>    & suffix,
    const std::map<std::string, float>& classify,
    const std::function<torch::Tensor(const std::string&, const torch::DeviceType&)> transform,
    const std::function<void(const torch::DeviceType&)> complete
) : device(lifuren::getDevice()) {
    if(!lifuren::file::exists(path) || !lifuren::file::is_directory(path)) {
        SPDLOG_WARN("数据集无效：{}", path);
        return;
    }
    auto iterator = std::filesystem::directory_iterator(std::filesystem::path(path));
    for(const auto& entry : iterator) {
        const auto entry_path = entry.path();
        const auto entry_name = entry_path.filename().string();
        const auto entry_iter = classify.find(entry_name);
        if(entry.is_directory() && entry_iter != classify.end()) {
            SPDLOG_DEBUG("加载分类：{}", entry_path.string());
            std::vector<std::string> files;
            lifuren::file::list_file(files, entry_path.string(), suffix);
            for(const auto& file : files) {
                SPDLOG_DEBUG("加载文件：{}", file);
                auto tensor = transform(file, this->device);
                if(tensor.numel() == 0) {
                    continue;
                }
                this->features.push_back(std::move(tensor));
            }
            std::vector<float> label;
            label.resize(classify.size(), 0.0F);
            label[entry_iter->second] = 1.0F;
            this->labels.resize(this->features.size(), torch::from_blob(label.data(), { static_cast<int>(label.size()) }, torch::kFloat32).clone().to(this->device));
        } else {
            SPDLOG_DEBUG("忽略目录：{}", entry_path.string());
        }
    }
    if(complete) {
        complete(this->device);
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
