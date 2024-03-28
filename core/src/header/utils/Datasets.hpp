/**
 * 文件数据集
 * 
 * @author acgist
 */
#pragma once

#include <map>
#include <string>
#include <vector>
#include <functional>

#include "torch/torch.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "./Files.hpp"
#include "../Logger.hpp"

namespace lifuren  {
namespace datasets {

// TODO: 支持文件列表

/**
 * 文件数据集
 * 
 * ./类型1/文件列表
 * ./类型2/文件列表
 * ...
 */
class FileDataset : public torch::data::Dataset<FileDataset> {

private:
    int label = 0;
    std::vector<int> labels;
    std::vector<std::string> paths;

public:
    std::map<std::string, int> nameLabel;
    std::map<int, std::string> labelName;
    std::function<torch::Tensor(const std::string&)> fileTransform;

    FileDataset(
        const std::string& path,
        const std::vector<std::string>& exts,
        const std::function<torch::Tensor(const std::string&)> fileTransform
    );

    torch::optional<size_t> size() const override;
    torch::data::Example<> get(size_t index) override;

};

// TODO: 变形、截取
inline auto loadImageDataset(
    const int width,
    const int height,
    const size_t batch_size,
    const std::string& path,
    const std::string& image_type
) -> decltype(auto) {
    // 注意这里width、height不能传递引用
    auto dataset = lifuren::datasets::FileDataset(path, { image_type }, [
        width,
        height
    ](const std::string& path) -> torch::Tensor {
        cv::Mat image = cv::imread(path);
        cv::resize(image, image, cv::Size(width, height));
        torch::Tensor data_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({2, 0, 1});
        // TODO: 验证
        image.release();
        // 不做正则
        // auto data_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kFloat32) / 225.0;
        return data_tensor;
    }).map(torch::data::transforms::Stack<>());
    auto loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
    // return std::move(loader);
    return loader;
}

typedef std::result_of<decltype(&lifuren::datasets::loadImageDataset)(const int, const int, const size_t, const std::string&, const std::string&)>::type ImageDatasetType;
// using ImageDatasetType = std::result_of<decltype(&lifuren::datasets::loadImageDataset)(const int, const int, const size_t, const std::string&, const std::string&)>::type;

}
}