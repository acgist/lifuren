#include "../../header/utils/Datasets.hpp"

lifuren::datasets::FileDataset::FileDataset(
    const std::string& path,
    const std::vector<std::string>& exts,
    const std::function<torch::Tensor(const std::string&)> fileTransform
) : fileTransform(fileTransform) {
    if(!std::filesystem::exists(path) || !std::filesystem::is_regular_file(path)) {
        SPDLOG_DEBUG("目录无效：{} - {}", __func__, path);
        return;
    }
    auto iterator = std::filesystem::directory_iterator(std::filesystem::u8path(path));
    for(const auto& entry : iterator) {
        std::string filepath = entry.path().u8string();
        if(entry.is_directory()) {
            std::string filename = entry.path().filename().u8string();
            this->nameLabel[filename] = this->label;
            this->labelName[this->label] = filename;
            const uint64_t oldSize = this->paths.size();
            lifuren::files::listFiles(this->paths, entry.path().u8string(), exts);
            const uint64_t newSize = this->paths.size();
            for(uint64_t index = oldSize; index < newSize; ++index) {
                this->labels.push_back(this->label);
            }
            this->label++;
        } else {
            SPDLOG_DEBUG("忽略无效文件：{} - {}", __func__, filepath);
        }
    }
}

torch::optional<size_t> lifuren::datasets::FileDataset::size() const {
    return this->paths.size();
}

torch::data::Example<> lifuren::datasets::FileDataset::get(size_t index) {
    const std::string& pathRef = this->paths.at(index);
    torch::Tensor data_tensor  = this->fileTransform(pathRef);
    const int label = this->labels.at(index);
    torch::Tensor label_tensor = torch::full({1}, label);
    return { 
        data_tensor.clone(),
        label_tensor.clone()
    };
}
