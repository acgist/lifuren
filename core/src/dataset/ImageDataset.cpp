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
    size_t video_count = 0;
    auto dataset = lifuren::dataset::Dataset(
        batch_size,
        path,
        { ".mp4" },
        [width, height, &frame_count, &video_count] (
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
            const auto video_frame_count  = video.get(cv::CAP_PROP_FRAME_COUNT);
            const auto video_frame_width  = video.get(cv::CAP_PROP_FRAME_WIDTH);
            const auto video_frame_height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
            SPDLOG_DEBUG("加载视频文件开始：{} - {} - {} - {}", video_fps, video_frame_count, video_frame_width, video_frame_height);
            size_t index = 0;
            size_t frame = 0;
            double mean;
            cv::Mat diff;
            cv::Mat dst_frame; // 目标视频帧
            cv::Mat old_frame; // 上次视频帧
            cv::Mat src_frame; // 当前视频帧
            torch::Tensor zero = torch::zeros({ 3, height, width }).to(LFR_DTYPE).to(device);
            std::vector<torch::Tensor> labels_batch;
            std::vector<torch::Tensor> features_batch;
            while(video.read(src_frame)) {
                #if LFR_VIDEO_FRAME_STEP > 0
                if(++index % LFR_VIDEO_FRAME_STEP != 0) {
                    continue;
                }
                #else
                ++index;
                #endif
                lifuren::dataset::image::resize(src_frame, width, height);
                mean = cv::mean(src_frame)[0];
                if(mean <= 10.0) {
                    // 跳过黑屏或者暗屏
                    continue;
                }
                if(!old_frame.empty()) {
                    cv::absdiff(src_frame, old_frame, diff);
                    mean = cv::mean(diff)[0];
                    if(mean == 0) {
                        // 没有变化
                        continue;
                    } else if(mean > LFR_VIDEO_DIFF || labels_batch.size() >= LFR_VIDEO_FRAME_MAX) {
                        if(labels_batch.size() >= LFR_VIDEO_FRAME_MIN) {
                            SPDLOG_DEBUG("加载视频片段：{} - {}", labels_batch.size(), features_batch.size());
                            frame += labels_batch.size();
                            labels_batch  .push_back(zero);
                            features_batch.push_back(zero);
                            labels  .insert(labels  .end(), std::make_move_iterator(labels_batch  .begin()), std::make_move_iterator(labels_batch  .end()));
                            features.insert(features.end(), std::make_move_iterator(features_batch.begin()), std::make_move_iterator(features_batch.end()));
                        } else {
                            SPDLOG_DEBUG("丢弃视频片段：{} - {}", labels_batch.size(), features_batch.size());
                        }
                        labels_batch  .clear();
                        features_batch.clear();
                    } else {
                        #ifdef LFR_VIDEO_DIFF_FRAME
                        dst_frame = src_frame - old_frame;
                        #else
                        dst_frame = src_frame;
                        #endif
                        labels_batch  .push_back(lifuren::dataset::image::mat_to_tensor(dst_frame).clone().to(LFR_DTYPE).to(device));
                        features_batch.push_back(lifuren::dataset::image::mat_to_tensor(old_frame).clone().to(LFR_DTYPE).to(device));
                    }
                }
                old_frame = src_frame;
            }
            if(labels_batch.size() >= LFR_VIDEO_FRAME_MIN) {
                SPDLOG_DEBUG("加载视频片段：{} - {}", labels_batch.size(), features_batch.size());
                frame += labels_batch.size();
                labels_batch  .push_back(zero);
                features_batch.push_back(zero);
                labels  .insert(labels  .end(), std::make_move_iterator(labels_batch  .begin()), std::make_move_iterator(labels_batch  .end()));
                features.insert(features.end(), std::make_move_iterator(features_batch.begin()), std::make_move_iterator(features_batch.end()));
            } else {
                SPDLOG_DEBUG("丢弃视频片段：{} - {}", labels_batch.size(), features_batch.size());
            }
            labels_batch  .clear();
            features_batch.clear();
            SPDLOG_DEBUG("加载视频文件完成：{} -> {} - {} / {}", video_count, file, frame, index);
            video.release();
            ++video_count;
            frame_count += frame;
        },
        nullptr,
        true
    ).map(torch::data::transforms::Stack<>());
    SPDLOG_DEBUG("视频数据集加载完成：{} - {}", video_count, frame_count);
    torch::data::DataLoaderOptions options(batch_size);
    options.drop_last(true);
    return torch::data::make_data_loader<LFT_SEQ_SAMPLER>(std::move(dataset), options);
}
