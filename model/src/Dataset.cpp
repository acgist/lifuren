#include "lifuren/Dataset.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"

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

bool lifuren::dataset::allDatasetPreprocessing(
    const std::string& path,
    const std::string& model_name,
    std::function<bool(const std::string&, const std::string&, std::ofstream&, lifuren::thread::ThreadPool&)> preprocessing,
    bool model_base
) {
    std::vector<std::string> datasets = std::move(lifuren::dataset::allDataset(path));
    if(datasets.empty()) {
        SPDLOG_WARN("没有数据集：{}", path);
        return false;
    }
    lifuren::thread::ThreadPool pool;
    return std::all_of(datasets.begin(), datasets.end(), [&path, &pool, &model_name, model_base, &preprocessing](const auto& dataset) {
        std::ofstream stream;
        std::filesystem::path model_path;
        if(model_base) {
            model_path = lifuren::file::join({ path,    lifuren::config::LIFUREN_HIDDEN_FILE, model_name});
        } else {
            model_path = lifuren::file::join({ dataset, lifuren::config::LIFUREN_HIDDEN_FILE, model_name});
        }
        lifuren::file::createParent(model_path);
        stream.open(model_path, std::ios_base::app | std::ios_base::out | std::ios_base::binary);
        if(!stream.is_open()) {
            stream.close();
            SPDLOG_WARN("文件打开失败：{}", model_path.string());
            return false;
        }
        bool ret = preprocessing(path, dataset, stream, pool);
        pool.wait_finish();
        stream.close();
        return ret;
    });
}
