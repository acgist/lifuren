#include "lifuren/video/Video.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/video/VideoModel.hpp"
#include "lifuren/image/ImageDataset.hpp"

template<typename M>
std::tuple<bool, std::string> lifuren::video::VideoClient<M>::pred(const VideoParams& input) {
    const std::string output = lifuren::file::modify_filename(input.video, ".mp4", "gen");;
    cv::VideoWriter writer(
        output,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        LFR_VIDEO_FPS,
        cv::Size(LFR_VIDEO_WIDTH, LFR_VIDEO_HEIGHT)
    );
    if(!writer.isOpened()) {
        SPDLOG_WARN("输出视频打开失败：{}", output);
        writer.release();
        return { false, output };
    }
    torch::Tensor pred_tensor;
    const auto suffix = lifuren::file::fileSuffix(input.video);
    if(suffix == ".mp4") {
        cv::VideoCapture reader(input.video);
        if(!reader.isOpened()) {
            SPDLOG_WARN("输入视频打开失败：{}", input.video);
            writer.release();
            reader.release();
            return { false, output };
        }
        cv::Mat frame;
        while(reader.read(frame)) {
            auto frame_feature = lifuren::image::feature(frame, LFR_VIDEO_WIDTH, LFR_VIDEO_HEIGHT, this->model->device);
            pred_tensor = this->model->pred(frame_feature);
            lifuren::image::resize(frame, LFR_VIDEO_WIDTH, LFR_VIDEO_HEIGHT);
            writer.write(frame);
        }
        reader.release();
    } else if(suffix == ".jpg" || suffix == ".png" || suffix == ".jpeg") {
        auto frame = cv::imread(input.video);
        if(frame.empty()) {
            SPDLOG_WARN("输入图片打开失败：{}", input.video);
            writer.release();
            return { false, output };
        }
        auto frame_feature = lifuren::image::feature(frame, LFR_VIDEO_WIDTH, LFR_VIDEO_HEIGHT, this->model->device);
        pred_tensor = this->model->pred(frame_feature);
        lifuren::image::resize(frame, LFR_VIDEO_WIDTH, LFR_VIDEO_HEIGHT);
        writer.write(frame);
    } else {
        SPDLOG_WARN("不支持的文件格式：{}", suffix);
        writer.release();
        return { false, output };
    }
    cv::Mat pred_frame(LFR_VIDEO_HEIGHT, LFR_VIDEO_WIDTH, CV_8UC3);
    for(int i = 0; i < LFR_VIDEO_PRED_FPS; ++i) {
        pred_tensor = this->model->pred(pred_tensor);
        lifuren::image::tensor_to_mat(pred_frame, pred_tensor);
        writer.write(pred_frame);
    }
    writer.release();
    return { true, output };
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
