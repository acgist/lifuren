#include "../../../header/nn/NNDataset.hpp"

lifuren::NNDataset::NNDataset(
    const std::string& path,
    torch::data::datasets::MNIST::Mode mode
) : mnistDataset(torch::data::datasets::MNIST(path, mode)) {
    
}
