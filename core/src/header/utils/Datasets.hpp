/**
 * 文件数据集
 * 
 * @author acgist
 */
#pragma once

#include <map>
#include <vector>
#include <string>

#include "torch/torch.h"

#include "./Files.hpp"
#include "../Logger.hpp"

namespace lifuren  {
namespace datasets {

/**
 * 文件数据转换
 */
class FileTransform {

public:
    torch::Tensor transform(const std::string& path);

};

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
    FileTransform fileTransform;
    std::map<std::string, int> nameLabel;
    std::map<int, std::string> labelName;

public:
    FileDataset(
        const std::string& path,
        const std::vector<std::string>& exts,
        const FileTransform& fileTransform
    );

public:
    torch::optional<size_t> size() const override;
    torch::data::Example<> get(size_t index) override;

};

}
}