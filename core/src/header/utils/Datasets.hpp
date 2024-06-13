/**
 * 文件数据集
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_UTILS_DATASETS_HPP
#define LFR_HEADER_CORE_UTILS_DATASETS_HPP

#include <map>
#include <string>
#include <vector>
#include <functional>

#include "torch/torch.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

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
    // 文件标签
    std::vector<int> labels;
    // 文件路径
    std::vector<std::string> paths;
    // 文件转换
    std::function<torch::Tensor(const std::string&)> fileTransform = nullptr;

public:
    /**
     * @param path          数据路径
     * @param exts          文件后缀
     * @param mapping       标签映射
     * @param fileTransform 文件转换
     */
    FileDataset(
        const std::string& path,
        const std::vector<std::string>& exts,
        const std::map<std::string, int>& mapping,
        const std::function<torch::Tensor(const std::string&)> fileTransform = nullptr
    );

public:
    /**
     * @return 数据集大小
     */
    torch::optional<size_t> size() const override;
    /**
     * @param index 文件索引
     * 
     * @return Tensor
     */
    torch::data::Example<> get(size_t index) override;

};

/**
 * 图片数据集
 * 
 * @param width          图片宽度
 * @param height         图片高度
 * @param batch_size     批次大小
 * @param path           图片路径
 * @param image_type     图片格式
 * @param mapping        标签映射
 * @param imageTransform 图片转换
 */
inline auto loadImageFileDataset(
    const int width,
    const int height,
    const size_t batch_size,
    const std::string& path,
    const std::string& image_type,
    const std::map<std::string, int>& mapping,
    const std::function<void(const cv::Mat&)> imageTransform = nullptr
) -> decltype(auto) {
    auto dataset = lifuren::datasets::FileDataset(path, {
        image_type
    }, mapping, [
        width,
        height,
        imageTransform
    ](const std::string& path) -> torch::Tensor {
        cv::Mat image = cv::imread(path);
        cv::resize(image, image, cv::Size(width, height));
        if(imageTransform != nullptr) {
            imageTransform(image);
        }
        torch::Tensor data_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({2, 0, 1});
        image.release();
        return data_tensor;
    }).map(torch::data::transforms::Stack<>());
    auto loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batch_size);
    return loader;
}

using ImageFileDataset = std::result_of<decltype(&lifuren::datasets::loadImageFileDataset)(
    const int,
    const int,
    const size_t,
    const std::string&,
    const std::string&,
    const std::map<std::string, int>&,
    const std::function<void(const cv::Mat&)>
)>::type;

}
}

#endif // LFR_HEADER_CORE_UTILS_DATASETS_HPP
