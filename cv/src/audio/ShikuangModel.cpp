#include "lifuren/audio/ComposeModel.hpp"

#include "lifuren/File.hpp"

lifuren::ShikuangModuleImpl::ShikuangModuleImpl() {
}

lifuren::ShikuangModuleImpl::~ShikuangModuleImpl() {
}

torch::Tensor lifuren::ShikuangModuleImpl::forward(torch::Tensor input) {
    // TODO: 实现
    return input;
}

lifuren::ShikuangModel::ShikuangModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::ShikuangModel::~ShikuangModel() {
}

bool lifuren::ShikuangModel::defineDataset() {
    // TODO：实现
    return true;
}

void lifuren::ShikuangModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    // TODO: 实现
}
