/**
 * 图片数据集
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_IMAGE_DATASET_HPP
#define LFR_HEADER_CV_IMAGE_DATASET_HPP

#include <cstdint>
#include <cstdlib>

#include "spdlog/spdlog.h"

#include "nlohmann/json.hpp"

#include "lifuren/File.hpp"
#include "lifuren/String.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/image/Image.hpp"

namespace lifuren::dataset {

namespace image {

extern torch::Tensor feature(const int& width, const int& height, const std::string& file, const torch::DeviceType& type);

inline torch::Tensor feature(char* data, const int& width, const int& height, const torch::DeviceType& type) {
    return torch::from_blob(data, { height, width, 3 }, torch::kByte).permute({2, 0, 1}).to(torch::kF32).div(255.0).clone().to(type);
}

} // END OF image


inline auto loadImageFileGANDataset(
    const int& width,
    const int& height,
    const size_t& batch_size,
    const std::string& path
) -> decltype(auto) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        ".json",
        { ".jpg", ".png", ".jpeg" },
        [width, height] (const std::string& image_file, const std::string& label_file, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& device) -> void {
            const std::string content = std::move(lifuren::file::loadFile(label_file));
            if(content.empty()) {
                SPDLOG_WARN("图片文件标记无效：{}", label_file);
                return;
            }
            const nlohmann::json prompts = std::move(nlohmann::json::parse(content));
            auto vector = std::move(lifuren::string::embedding(prompts.get<std::vector<std::string>>()));
            if(vector.empty()) {
                SPDLOG_WARN("图片文件标记无效：{}", label_file);
                return;
            }
            labels.push_back(torch::from_blob(vector.data(), { static_cast<int>(vector.size()) }, torch::kFloat32).clone().to(device));
            features.push_back(std::move(lifuren::dataset::image::feature(width, height, image_file, device)));
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

using ImageFileGANDatasetLoader = std::invoke_result<
    decltype(&lifuren::dataset::loadImageFileGANDataset),
    const int&,
    const int&,
    const size_t&,
    const std::string&
>::type;

inline auto loadImageFileStyleDataset(
    const int& width,
    const int& height,
    const size_t& batch_size,
    const std::string& path
) -> decltype(auto) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        "source",
        "target",
        { ".jpg", ".png", ".jpeg" },
        [width, height] (const std::string& source, const std::string& target, std::vector<torch::Tensor>& labels, std::vector<torch::Tensor>& features, const torch::DeviceType& device) -> void {
            labels.push_back(std::move(lifuren::dataset::image::feature(width, height, target, device)));
            features.push_back(std::move(lifuren::dataset::image::feature(width, height, source, device)));
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

using ImageFileStyleDatasetLoader = std::invoke_result<
    decltype(&lifuren::dataset::loadImageFileStyleDataset),
    const int&,
    const int&,
    const size_t&,
    const std::string&
>::type;

/**
 * @param width      图片宽度
 * @param height     图片高度
 * @param batch_size 批次大小
 * @param path       图片路径
 * @param classify   标签映射
 * 
 * @return 图片数据集
 */
inline auto loadImageFileClassifyDataset(
    const int& width,
    const int& height,
    const size_t batch_size,
    const std::string& path,
    const std::map<std::string, float>& classify
) -> decltype(auto) {
    auto dataset = lifuren::dataset::FileDataset(
        path,
        { ".jpg", ".png", ".jpeg" },
        classify,
        [width, height] (const std::string& file, const torch::DeviceType& device) -> torch::Tensor {
            return lifuren::dataset::image::feature(width, height, file, device);
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
}

using ImageFileClassifyDatasetLoader = std::invoke_result<
    decltype(&lifuren::dataset::loadImageFileClassifyDataset),
    const int&,
    const int&,
    const size_t&,
    const std::string&,
    const std::map<std::string, float>&
>::type;

} // END OF lifuren::dataset

#endif // END OF LFR_HEADER_CV_IMAGE_DATASET_HPP
