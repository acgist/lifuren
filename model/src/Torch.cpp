#include "lifuren/Torch.hpp"

void lifuren::setDevice(torch::DeviceType& type) {
    if(torch::cuda::is_available()) {
        type = torch::DeviceType::CUDA;
    } else {
        type = torch::DeviceType::CPU;
    }
}
