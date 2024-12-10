#include "lifuren/video/ActModel.hpp"

#include "lifuren/File.hpp"

lifuren::TangxianzuModuleImpl::TangxianzuModuleImpl() {
}

lifuren::TangxianzuModuleImpl::~TangxianzuModuleImpl() {
}

torch::Tensor lifuren::TangxianzuModuleImpl::forward(torch::Tensor input) {
    // TODO: 实现
    return input;
}

lifuren::TangxianzuModel::TangxianzuModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::TangxianzuModel::~TangxianzuModel() {
}

bool lifuren::TangxianzuModel::defineDataset() {
    // TODO：实现
    return true;
}

void lifuren::TangxianzuModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    // TODO: 实现
}
