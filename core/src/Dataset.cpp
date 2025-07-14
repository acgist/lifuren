#include "lifuren/Dataset.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "torch/cuda.h"
#include "torch/data.h"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"

torch::DeviceType lifuren::get_device() {
    if(torch::cuda::is_available()) {
        return torch::DeviceType::CUDA;
    } else {
        return torch::DeviceType::CPU;
    }
}

lifuren::dataset::Dataset::Dataset(
    size_t batch_size,
    std::vector<torch::Tensor>& labels,
    std::vector<torch::Tensor>& features
) : batch_size(batch_size), device(lifuren::get_device()), labels(std::move(labels)), features(std::move(features)) {
}

lifuren::dataset::Dataset::Dataset(
    size_t batch_size,
    const std::string& path,
    const std::vector<std::string>& suffix,
    const Transform transform
) : batch_size(batch_size), device(lifuren::get_device()) {
    if(!lifuren::file::exists(path) || !lifuren::file::is_directory(path)) {
        SPDLOG_WARN("数据集无效：{}", path);
        return;
    }
    std::vector<std::string> files;
    lifuren::file::list_file(files, path, suffix);
    for(const auto& file : files) {
        SPDLOG_DEBUG("加载文件：{}", file);
        transform(file, this->labels, this->features, this->device);
    }
}

lifuren::dataset::Dataset::~Dataset() {
}

torch::optional<size_t> lifuren::dataset::Dataset::size() const {
    return this->labels.size();
}

torch::data::Example<> lifuren::dataset::Dataset::get(size_t index) {
    return {
        this->features[index],
        this->labels  [index]
    };
}

torch::Tensor lifuren::dataset::image::pose(cv::Mat& pose, const cv::Mat& prev, const cv::Mat& next) {
    cv::Mat diff = next - prev;
    cv::resize(diff, diff, cv::Size(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT), 0, 0, cv::INTER_AREA);
    cv::threshold(diff, diff, LFR_VIDEO_BLACK_MEAN, 255, cv::THRESH_BINARY);
    std::vector<cv::Mat> channels;
    cv::split(diff, channels);
    pose = channels[0] + channels[1] + channels[2]; // ? / 3 ?
    cv::resize(pose, pose, cv::Size(LFR_VIDEO_POSE_WIDTH, LFR_VIDEO_POSE_HEIGHT), 0, 0, cv::INTER_AREA);
    // 不二值化更加丰富
    // cv::threshold(pose, pose, LFR_VIDEO_BLACK_MEAN, 255, cv::THRESH_BINARY);
    return torch::from_blob(pose.data, { pose.rows, pose.cols }, torch::kByte).to(torch::kFloat32).div(255.0).mul(2.0).sub(1.0).contiguous();
}

void lifuren::dataset::image::resize(cv::Mat& image, const int width, const int height) {
    #ifdef LFR_VIDEO_FILL_FRAME
    const int cols = image.cols;
    const int rows = image.rows;
    const double ws = 1.0 * cols / width;
    const double hs = 1.0 * rows / height;
    const double scale = std::max(ws, hs);
    const int w = std::max(static_cast<int>(width  * scale), cols);
    const int h = std::max(static_cast<int>(height * scale), rows);
    cv::Mat result = cv::Mat::zeros(h, w, CV_8UC3);
    image.copyTo(result(cv::Rect(0, 0, cols, rows)));
    if(scale < 1.0) {
        cv::resize(result, image, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    } else {
        cv::resize(result, image, cv::Size(width, height), 0, 0, cv::INTER_AREA);
    }
    #else
    const int cols = image.cols;
    const int rows = image.rows;
    const double ws = 1.0 * cols / width;
    const double hs = 1.0 * rows / height;
    const double scale = std::min(ws, hs);
    const int w = std::min(static_cast<int>(width  * scale), cols);
    const int h = std::min(static_cast<int>(height * scale), rows);
    image = image(cv::Rect((cols - w) / 2, (rows - h) / 2, w, h));
    if(scale < 1.0) {
        cv::resize(image, image, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    } else {
        cv::resize(image, image, cv::Size(width, height), 0, 0, cv::INTER_AREA);
    }
    #endif
}

torch::Tensor lifuren::dataset::image::mat_to_tensor(const cv::Mat& image) {
    if(image.empty()) {
        return {};
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    return torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }).to(torch::kFloat32).div(255.0).mul(2.0).sub(1.0).contiguous();
}

void lifuren::dataset::image::tensor_to_mat(cv::Mat& image, const torch::Tensor& tensor) {
    if(image.empty()) {
        return;
    }
    if(tensor.dim() == 3) {
        auto image_tensor = tensor.permute({ 1, 2, 0 }).add(1.0).mul(255.0).div(2.0).to(torch::kByte).contiguous();
        std::memcpy(image.data, reinterpret_cast<char*>(image_tensor.data_ptr()), image.total() * image.elemSize());
    } else {
        // int N  = tensor.size(0);
        int C  = tensor.size(1);
        int H  = tensor.size(2);
        int W  = tensor.size(3);
        int HN = image.rows / H;
        int WN = image.cols / W;
        auto image_tensor = tensor
            .reshape({ HN, WN, C, H, W   }) // N C H W     -> HN WN C H W
            .permute({ 2, 0, 3, 1, 4     }) // HN WN C H W -> C HN H WN W
            .reshape({ C, HN * H, WN * W }) // C HN H WN W -> C H W
            .permute({ 1, 2, 0           }) // C H W       -> H W C
            .add(1.0).mul(255.0).div(2.0).to(torch::kByte).contiguous();
        std::memcpy(image.data, reinterpret_cast<char*>(image_tensor.data_ptr()), image.total() * image.elemSize());
    }
    if(image.channels() == 3) {
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    }
}

lifuren::dataset::RndDatasetLoader lifuren::dataset::image::loadWudaoziDatasetLoader(const int width, const int height, const size_t batch_size, const std::string& path) {
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
            const torch::DeviceType   & /*device*/
        ) {
            cv::VideoCapture video;
            video.open(file);
            if(!video.isOpened()) {
                SPDLOG_WARN("加载视频文件失败：{}", file);
                return;
            }
            SPDLOG_DEBUG(
                "加载视频文件开始：{} - {} - {} - {}",
                video.get(cv::CAP_PROP_FPS),
                video.get(cv::CAP_PROP_FRAME_COUNT),
                video.get(cv::CAP_PROP_FRAME_WIDTH),
                video.get(cv::CAP_PROP_FRAME_HEIGHT)
            );
            size_t index = 0;
            size_t frame = 0;
            double mean;
            cv::Mat diff;
            cv::Mat prev_frame;
            cv::Mat next_frame;
            cv::Mat pose(LFR_VIDEO_POSE_HEIGHT, LFR_VIDEO_POSE_WIDTH, CV_8UC1);
            std::vector<torch::Tensor> pose_tensor;
            std::vector<torch::Tensor> frame_tensor;
            while(video.read(next_frame)) {
                #if LFR_VIDEO_FRAME_STEP > 0
                if(++index % LFR_VIDEO_FRAME_STEP != 0) {
                    continue;
                }
                #else
                ++index;
                #endif
                lifuren::dataset::image::resize(next_frame, width, height);
                mean = cv::mean(next_frame)[0];
                if(mean <= LFR_VIDEO_BLACK_MEAN) {
                    // 跳过黑屏或者暗屏
                    continue;
                }
                if(!prev_frame.empty()) {
                    cv::absdiff(next_frame, prev_frame, diff);
                    mean = cv::mean(diff)[0];
                    if(mean == 0) {
                        // 没有变化
                        continue;
                    } else if(mean > LFR_VIDEO_DIFF || frame_tensor.size() >= LFR_VIDEO_FRAME_MAX) {
                        if(frame_tensor.size() >= LFR_VIDEO_FRAME_MIN) {
                            SPDLOG_DEBUG("加载视频片段：{}", frame_tensor.size());
                            frame += frame_tensor.size();
                            auto feature = torch::stack(frame_tensor, 0).clone();
                            for(size_t i = 0; i < frame_tensor.size() - 1; ++i) {
                                features.push_back(feature.slice(0, i, i + 2));
                                labels  .push_back(torch::stack({
                                    torch::ones({ LFR_VIDEO_POSE_HEIGHT, LFR_VIDEO_POSE_WIDTH }).mul(i + 1),
                                    pose_tensor[i]
                                }, 0));
                            }
                        } else {
                            SPDLOG_DEBUG("丢弃视频片段：{}", frame_tensor.size());
                        }
                        pose_tensor .clear();
                        frame_tensor.clear();
                    } else {
                        pose_tensor .push_back(lifuren::dataset::image::pose(pose, prev_frame, next_frame));
                        frame_tensor.push_back(lifuren::dataset::image::mat_to_tensor(prev_frame));
                    }
                }
                prev_frame = next_frame;
            }
            if(frame_tensor.size() >= LFR_VIDEO_FRAME_MIN) {
                SPDLOG_DEBUG("加载视频片段：{}", frame_tensor.size());
                frame += frame_tensor.size();
                auto feature = torch::stack(frame_tensor, 0).clone();
                for(size_t i = 0; i < frame_tensor.size() - 1; ++i) {
                    features.push_back(feature.slice(0, i, i + 2));
                    labels  .push_back(torch::stack({
                        torch::ones({ LFR_VIDEO_POSE_HEIGHT, LFR_VIDEO_POSE_WIDTH }).mul(i + 1),
                        pose_tensor[i]
                    }, 0));
                }
            } else {
                SPDLOG_DEBUG("丢弃视频片段：{}", frame_tensor.size());
            }
            pose_tensor .clear();
            frame_tensor.clear();
            SPDLOG_DEBUG("加载视频文件完成：{} -> {} - {} / {}", video_count, file, frame, index);
            video.release();
            ++video_count;
            frame_count += frame;
        }
    ).map(torch::data::transforms::Stack<>());
    SPDLOG_DEBUG("视频数据集加载完成：{} - {}", video_count, frame_count);
    torch::data::DataLoaderOptions options(batch_size);
    options.drop_last(true);
    return torch::data::make_data_loader<LFT_RND_SAMPLER>(std::move(dataset), options);
}
