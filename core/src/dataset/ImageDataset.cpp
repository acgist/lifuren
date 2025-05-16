#include "lifuren/Dataset.hpp"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"

#include "opencv2/opencv.hpp"

void lifuren::dataset::image::resize(cv::Mat& image, const int width, const int height) {
    #ifdef LFR_VIDEO_FILL
    const int cols = image.cols;
    const int rows = image.rows;
    const double ws = 1.0 * cols / width;
    const double hs = 1.0 * rows / height;
    const double scale = std::max(ws, hs);
    const int w = std::max(static_cast<int>(width  * scale), cols);
    const int h = std::max(static_cast<int>(height * scale), rows);
    cv::Mat result = cv::Mat::zeros(h, w, CV_8UC3);
    image.copyTo(result(cv::Rect(0, 0, cols, rows)));
    cv::resize(result, image, cv::Size(width, height));
    #else
    const int cols = image.cols;
    const int rows = image.rows;
    const double ws = 1.0 * cols / width;
    const double hs = 1.0 * rows / height;
    const double scale = std::min(ws, hs);
    const int w = std::min(static_cast<int>(width  * scale), cols);
    const int h = std::min(static_cast<int>(height * scale), rows);
    image = image(cv::Rect((cols - w) / 2, (rows - h) / 2, w, h));
    cv::resize(image, image, cv::Size(width, height));
    #endif
}

torch::Tensor lifuren::dataset::image::mat_to_tensor(const cv::Mat& image) {
    if(image.empty()) {
        return {};
    }
    return torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({2, 0, 1}).to(torch::kFloat32).div(255.0);
}

void lifuren::dataset::image::tensor_to_mat(cv::Mat& image, const torch::Tensor& tensor) {
    if(image.empty()) {
        return;
    }
    auto image_tensor = tensor.permute({1, 2, 0}).mul(255.0).to(torch::kByte).contiguous();
    std::memcpy(image.data, reinterpret_cast<char*>(image_tensor.data_ptr()), image.total() * image.elemSize());
}

lifuren::dataset::SeqDatasetLoader lifuren::dataset::image::loadWudaoziDatasetLoader(const int width, const int height, const size_t batch_size, const std::string& path) {
    size_t frame_count = 0;
    auto dataset = lifuren::dataset::Dataset(
        batch_size,
        path,
        { ".mp4" },
        [width, height, &frame_count] (
            const std::string         & file,
            std::vector<torch::Tensor>& labels,
            std::vector<torch::Tensor>& features,
            const torch::DeviceType   & device
        ) {
            cv::VideoCapture video;
            video.open(file);
            if(!video.isOpened()) {
                SPDLOG_WARN("加载视频文件失败：{}", file);
                return;
            }
            const auto video_fps          = video.get(cv::CAP_PROP_FPS);
            const auto video_frame_type   = video.get(cv::CAP_PROP_FRAME_TYPE);
            const auto video_frame_count  = video.get(cv::CAP_PROP_FRAME_COUNT);
            const auto video_frame_width  = video.get(cv::CAP_PROP_FRAME_WIDTH);
            const auto video_frame_height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
            SPDLOG_DEBUG("加载视频文件开始：{} - {} - {} - {} - {}", video_fps, video_frame_type, video_frame_count, video_frame_width, video_frame_height);
            size_t index = 0;
            size_t frame = 0;
            cv::Mat diff;
            cv::Mat label;
            cv::Mat feature;
            torch::Tensor zero = torch::zeros({ 3, height, width }).to(device);
            std::vector<torch::Tensor> labels_local;
            std::vector<torch::Tensor> features_local;
            while(video.read(feature)) {
                #if LFR_VIDEO_FRAME_STEP > 0
                if(++index % LFR_VIDEO_FRAME_STEP != 0) {
                    continue;
                }
                #else
                ++index;
                #endif
                lifuren::dataset::image::resize(feature, width, height);
                double min = 0;
                double max = 0;
                cv::minMaxLoc(feature, &min, &max);
                if(max == 0 && min == 0) {
                    // 跳过黑屏
                    continue;
                }
                if(!label.empty()) {
                    cv::absdiff(feature, label, diff);
                    auto diff_mean = cv::mean(diff)[0];
                    if(diff_mean == 0) {
                        // 没有变化
                        continue;
                    } else if(diff_mean > LFR_VIDEO_DIFF || labels_local.size() >= LFR_VIDEO_FRAME_MAX) {
                        label = feature;
                        if(labels_local.size() >= LFR_VIDEO_FRAME_MIN) {
                            SPDLOG_DEBUG("加载视频片段：{} - {}", labels_local.size(), features_local.size());
                            frame += labels_local.size();
                            labels_local  .push_back(zero);
                            features_local.push_back(zero);
                            labels  .insert(labels  .end(), std::make_move_iterator(labels_local  .begin()), std::make_move_iterator(labels_local  .end()));
                            features.insert(features.end(), std::make_move_iterator(features_local.begin()), std::make_move_iterator(features_local.end()));
                        } else {
                            SPDLOG_DEBUG("丢弃视频片段：{} - {}", labels_local.size(), features_local.size());
                        }
                        labels_local  .clear();
                        features_local.clear();
                        continue;
                    } else {
                        // -
                    }
                }
                if(!label.empty()) {
                    labels_local  .push_back(lifuren::dataset::image::mat_to_tensor(label  ).clone().to(device));
                    features_local.push_back(lifuren::dataset::image::mat_to_tensor(feature).clone().to(device));
                }
                label = feature;
            }
            if(labels_local.size() >= LFR_VIDEO_FRAME_MIN) {
                SPDLOG_DEBUG("加载视频片段：{} - {}", labels_local.size(), features_local.size());
                frame += labels_local.size();
                labels_local  .push_back(zero);
                features_local.push_back(zero);
                labels  .insert(labels  .end(), std::make_move_iterator(labels_local  .begin()), std::make_move_iterator(labels_local  .end()));
                features.insert(features.end(), std::make_move_iterator(features_local.begin()), std::make_move_iterator(features_local.end()));
            } else {
                SPDLOG_DEBUG("丢弃视频片段：{} - {}", labels_local.size(), features_local.size());
            }
            frame_count += frame;
            labels_local  .clear();
            features_local.clear();
            SPDLOG_DEBUG("加载视频文件完成：{} - {} / {}", file, frame, index);
            video.release();
        },
        nullptr,
        true
    ).map(torch::data::transforms::Stack<>());
    SPDLOG_DEBUG("视频数据集加载完成：{}", frame_count);
    torch::data::DataLoaderOptions options(batch_size);
    options.drop_last(true);
    // TODO: gru
    return torch::data::make_data_loader<LFT_SEQ_SAMPLER>(std::move(dataset), options);
}
