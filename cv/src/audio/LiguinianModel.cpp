#include "lifuren/audio/ComposeModel.hpp"

#include "lifuren/File.hpp"

lifuren::LiguinianModuleImpl::LiguinianModuleImpl() {
}

lifuren::LiguinianModuleImpl::~LiguinianModuleImpl() {
}

torch::Tensor lifuren::LiguinianModuleImpl::forward(torch::Tensor input) {
    // TODO: 实现
    return input;
}

lifuren::LiguinianModel::LiguinianModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::LiguinianModel::~LiguinianModel() {
}

bool lifuren::LiguinianModel::defineDataset() {
    // TODO：实现
    return true;
}

void lifuren::LiguinianModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    // TODO: 实现
}
