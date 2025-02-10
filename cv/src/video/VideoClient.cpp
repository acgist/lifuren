#include "lifuren/video/Video.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "lifuren/video/VideoModel.hpp"
#include "lifuren/image/ImageDataset.hpp"

template<typename M>
std::tuple<bool, std::string> lifuren::video::VideoClient<M>::pred(const VideoParams& input) {
    cv::VideoWriter writer(
        input.output,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        LFR_VIDEO_FPS,
        cv::Size(LFR_VIDEO_WIDTH, LFR_VIDEO_HEIGHT)
    );
    if(!writer.isOpened()) {
        SPDLOG_WARN("输出视频打开失败：{}", input.output);
        writer.release();
        return { false, input.output };
    }
    torch::Tensor pred_tensor;
    if(!input.video.empty()) {
        cv::VideoCapture reader(input.video);
        if(!reader.isOpened()) {
            SPDLOG_WARN("输入视频打开失败：{}", input.video);
            reader.release();
            return { false, input.output };
        }
        cv::Mat frame;
        while(reader.read(frame)) {
            auto frame_feature = lifuren::image::feature(frame, LFR_VIDEO_WIDTH, LFR_VIDEO_HEIGHT, this->model->device);
            pred_tensor = this->model->pred(frame_feature);
            lifuren::image::resize(frame, LFR_VIDEO_WIDTH, LFR_VIDEO_HEIGHT);
            writer.write(frame);
        }
        reader.release();
    } else if(!input.image.empty()) {
        auto frame = cv::imread(input.image);
        auto frame_feature = lifuren::image::feature(frame, LFR_VIDEO_WIDTH, LFR_VIDEO_HEIGHT, this->model->device);
        pred_tensor = this->model->pred(frame_feature);
        lifuren::image::resize(frame, LFR_VIDEO_WIDTH, LFR_VIDEO_HEIGHT);
        writer.write(frame);
    } else {
        SPDLOG_WARN("没有输入文件");
        writer.release();
        return { false, input.output };
    }
    cv::Mat pred_frame(LFR_VIDEO_HEIGHT, LFR_VIDEO_WIDTH, CV_8UC3);
    for(int i = 0; i < LFR_VIDEO_PRED_FPS; ++i) {
        pred_tensor = this->model->pred(pred_tensor);
        lifuren::image::tensor_to_mat(pred_frame, pred_tensor);
        writer.write(pred_frame);
    }
    writer.release();
    return { true, input.output };
};

std::unique_ptr<lifuren::video::VideoModelClient> lifuren::video::getVideoClient(
    const std::string& model
) {
    if(model == lifuren::config::CONFIG_VIDEO_WUDAOZI) {
        return std::make_unique<lifuren::video::VideoClient<WudaoziModel>>();
    } else {
        return nullptr;
    }
}
