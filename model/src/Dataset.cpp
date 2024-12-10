#include "lifuren/Dataset.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"

#include "spdlog/spdlog.h"

std::vector<std::string> lifuren::dataset::allDataset(const std::string& path) {
    std::vector<std::string> ret;
    ret.reserve(3);
    const auto train_path = lifuren::file::join({ path, lifuren::config::DATASET_TRAIN });
    const auto val_path   = lifuren::file::join({ path, lifuren::config::DATASET_VAL   });
    const auto test_path  = lifuren::file::join({ path, lifuren::config::DATASET_TEST  });
    if(std::filesystem::exists(train_path)) {
        ret.push_back(train_path.string());
    }
    if(std::filesystem::exists(val_path)) {
        ret.push_back(val_path.string());
    }
    if(std::filesystem::exists(test_path)) {
        ret.push_back(test_path.string());
    }
    return ret;
}

bool lifuren::dataset::allDatasetPreprocessing(const std::string& path, std::function<bool(const std::string&)> preprocessing) {
    std::vector<std::string> datasets = std::move(lifuren::dataset::allDataset(path));
    if(datasets.empty()) {
        SPDLOG_WARN("没有数据集：{}", path);
        return false;
    }
    return std::all_of(datasets.begin(), datasets.end(), [&preprocessing](const auto& dataset) {
        return preprocessing(dataset);
    });
}
