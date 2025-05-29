#include "lifuren/Image.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/ImageModel.hpp"

namespace lifuren::image {

template<typename M>
using ImageModelClientImpl = ModelClientImpl<lifuren::config::ModelParams, std::string, std::string, M>;

template<typename M>
class ImageClient : public ImageModelClientImpl<M> {

public:
    std::tuple<bool, std::string> pred(const std::string& input) override;
    std::tuple<bool, std::string> predImage(const std::string& input);
    std::tuple<bool, std::string> predVideo(const std::string& input);

};

};

template<>
std::tuple<bool, std::string> lifuren::image::ImageClient<lifuren::image::WudaoziModel>::predImage(const std::string& input) {
    auto image = cv::imread(input);
    if(image.empty()) {
        return { false, {} };
    }
    const auto output = lifuren::file::modify_filename(input, ".mp4", "gen");
    cv::VideoWriter writer(output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), LFR_VIDEO_FPS, cv::Size(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT));
    if(!writer.isOpened()) {
        SPDLOG_WARN("视频文件打开失败：{}", output);
        return { false, output };
    }
    int index = 0;
    lifuren::dataset::image::resize(image, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
    auto tensor = lifuren::dataset::image::mat_to_tensor(image).unsqueeze(0).to(LFR_DTYPE).to(lifuren::getDevice());
    for(int i = 0; i < LFR_VIDEO_FRAME_SIZE; ++i) {
        #ifdef LFR_VIDEO_DIFF_FRAME
        auto result = this->model->pred(tensor).add(tensor);
        #else
        auto result = this->model->pred(tensor);
        #endif
        lifuren::dataset::image::tensor_to_mat(image, result.squeeze().to(torch::kCPU).to(torch::kFloat32));
        writer.write(image);
        tensor = result;
    }
    writer.release();
    return { true, output };
}

template<>
std::tuple<bool, std::string> lifuren::image::ImageClient<lifuren::image::WudaoziModel>::predVideo(const std::string& input) {
    cv::VideoCapture video(input);
    if(!video.isOpened()) {
        return { false, {} };
    }
    const auto output = lifuren::file::modify_filename(input, ".mp4", "gen");
    cv::VideoWriter writer(output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), LFR_VIDEO_FPS, cv::Size(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT));
    if(!writer.isOpened()) {
        SPDLOG_WARN("视频文件打开失败：{}", output);
        return { false, output };
    }
    int index = 0;
    cv::Mat image;
    while(video.read(image)) {
        lifuren::dataset::image::resize(image, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
        auto tensor = lifuren::dataset::image::mat_to_tensor(image).unsqueeze(0).to(LFR_DTYPE).to(lifuren::getDevice());
        auto result = this->model->pred(tensor);
        // auto result = this->model->pred(tensor).add(tensor);
        lifuren::dataset::image::tensor_to_mat(image, result.squeeze().to(torch::kCPU).to(torch::kFloat32));
        writer.write(image);
    }
    video.release();
    writer.release();
    return { true, output };
}

template<>
std::tuple<bool, std::string> lifuren::image::ImageClient<lifuren::image::WudaoziModel>::pred(const std::string& input) {
    if(!this->model) {
        return { false, {} };
    }
    auto suffix = lifuren::file::file_suffix(input);
    if(suffix == ".mp4") {
        return this->predVideo(input);
    } else {
        return this->predImage(input);
    }
}

std::unique_ptr<lifuren::image::ImageModelClient> lifuren::image::getImageClient(const std::string& model) {
    if(model == "wudaozi") {
        return std::make_unique<lifuren::image::ImageClient<WudaoziModel>>();
    } else {
        return nullptr;
    }
}
