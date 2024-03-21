/**
 * 文件数据集
 * 
 * @author acgist
 */
#pragma once

#include <map>
#include <vector>
#include <string>
#include <functional>

#include "torch/torch.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "./Files.hpp"
#include "../Logger.hpp"

namespace lifuren  {
namespace datasets {

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
    std::function<torch::Tensor(const std::string&)> fileTransform;
    std::map<std::string, int> nameLabel;
    std::map<int, std::string> labelName;

    FileDataset(
        const std::string& path,
        const std::vector<std::string>& exts,
        const std::function<torch::Tensor(const std::string&)> fileTransform
    );

    torch::optional<size_t> size() const override;
    torch::data::Example<> get(size_t index) override;

};

// TODO：高宽参数
inline auto loadImageDataset(int batch_size, const std::string& path, const std::string& image_type) -> decltype(auto) {
    auto dataset = lifuren::datasets::FileDataset(path, { image_type }, [](const std::string& path) -> torch::Tensor {
        cv::Mat image = cv::imread(path);
        cv::resize(image, image, cv::Size(224, 224));
        torch::Tensor data_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({2, 0, 1});
        return data_tensor;
    }).map(torch::data::transforms::Stack<>());
    auto loader  = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
    return std::move(loader);
}

typedef std::result_of<decltype(&lifuren::datasets::loadImageDataset)(int, const std::string&, const std::string&)>::type ImageDatasetType;

}
}