#include "lifuren/image/PaintModel.hpp"

#include "lifuren/File.hpp"

lifuren::WudaoziModuleImpl::WudaoziModuleImpl() {
}

lifuren::WudaoziModuleImpl::~WudaoziModuleImpl() {
}

torch::Tensor lifuren::WudaoziModuleImpl::forward(torch::Tensor input) {
    // TODO: 实现
    return input;
}

lifuren::WudaoziModel::WudaoziModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::WudaoziModel::~WudaoziModel() {
}

bool lifuren::WudaoziModel::defineDataset() {
    // TODO：实现
    return true;
}

void lifuren::WudaoziModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    // TODO: 实现
}
