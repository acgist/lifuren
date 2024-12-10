#include "lifuren/video/ActModel.hpp"

#include "lifuren/File.hpp"

lifuren::GuanhanqingModuleImpl::GuanhanqingModuleImpl() {
}

lifuren::GuanhanqingModuleImpl::~GuanhanqingModuleImpl() {
}

torch::Tensor lifuren::GuanhanqingModuleImpl::forward(torch::Tensor input) {
    // TODO: 实现
    return input;
}

lifuren::GuanhanqingModel::GuanhanqingModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::GuanhanqingModel::~GuanhanqingModel() {
}

bool lifuren::GuanhanqingModel::defineDataset() {
    // TODO：实现
    return true;
}

void lifuren::GuanhanqingModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    // TODO: 实现
}
