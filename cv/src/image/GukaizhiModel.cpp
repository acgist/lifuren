#include "lifuren/image/PaintModel.hpp"

#include "lifuren/File.hpp"

lifuren::GukaizhiModuleImpl::GukaizhiModuleImpl() {
}

lifuren::GukaizhiModuleImpl::~GukaizhiModuleImpl() {
}

torch::Tensor lifuren::GukaizhiModuleImpl::forward(torch::Tensor input) {
    // TODO: 实现
    return input;
}

lifuren::GukaizhiModel::GukaizhiModel(lifuren::config::ModelParams params) : Model(params) {
}

lifuren::GukaizhiModel::~GukaizhiModel() {
}

bool lifuren::GukaizhiModel::defineDataset() {
    // TODO：实现
    return true;
}

void lifuren::GukaizhiModel::logic(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) {
    // TODO: 实现
}
