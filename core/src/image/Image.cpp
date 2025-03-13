#include "lifuren/image/Image.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/image/ImageModel.hpp"

std::tuple<bool, std::string> lifuren::image::ImageClient<lifuren::image::ChopinModel>::pred(const std::string& input) {
    return {};
}

std::tuple<bool, std::string> lifuren::image::ImageClient<lifuren::image::MozartModel>::pred(const std::string& input) {
    return {};
}

std::tuple<bool, std::string> lifuren::image::ImageClient<lifuren::image::WudaoziModel>::pred(const std::string& input) {
    const std::string output = lifuren::file::modify_filename(input, ".mp4", "gen");;
    torch::Tensor pred_tensor;
    const auto suffix = lifuren::file::fileSuffix(input);
    if(suffix == ".jpg" || suffix == ".png" || suffix == ".jpeg") {
        auto frame = cv::imread(input);
        if(frame.empty()) {
            SPDLOG_WARN("输入图片打开失败：{}", input);
            return { false, output };
        }
        lifuren::dataset::image::resize(frame, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
        auto frame_feature = lifuren::dataset::image::feature(frame, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
        pred_tensor = this->model->pred(frame_feature.to(this->model->device));
    } else {
        SPDLOG_WARN("不支持的文件格式：{}", suffix);
        return { false, output };
    }
    cv::Mat pred_frame(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, CV_8UC3);
    pred_tensor = this->model->pred(pred_tensor);
    lifuren::dataset::image::tensor_to_mat(pred_frame, pred_tensor);
    // cv::imwrite(); // TODO
    return { true, output };
}

std::unique_ptr<lifuren::image::ImageModelClient> lifuren::image::getImageClient(
    const std::string& model
) {
    if(model == "chopin") {
        return std::make_unique<lifuren::image::ImageClient<ChopinModel>>();
    } else if(model == "mozart") {
        return std::make_unique<lifuren::image::ImageClient<MozartModel>>();
    } else if(model == "wudaozi") {
        return std::make_unique<lifuren::image::ImageClient<WudaoziModel>>();
    } else {
        return nullptr;
    }
}
