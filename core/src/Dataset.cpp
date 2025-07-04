#include "lifuren/Dataset.hpp"

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Config.hpp"

std::vector<std::string> lifuren::dataset::allDataset(const std::string& path) {
    std::vector<std::string> ret;
    ret.reserve(3);
    const auto train_path = lifuren::file::join({ path, lifuren::config::DATASET_TRAIN });
    const auto val_path   = lifuren::file::join({ path, lifuren::config::DATASET_VAL   });
    const auto test_path  = lifuren::file::join({ path, lifuren::config::DATASET_TEST  });
    if(std::filesystem::exists(train_path)) {
        ret.push_back(train_path.string());
    } else {
        SPDLOG_DEBUG("无效的训练数据集：{}", train_path.string());
    }
    if(std::filesystem::exists(val_path)) {
        ret.push_back(val_path.string());
    } else {
        SPDLOG_DEBUG("无效的验证训练集：{}", val_path.string());
    }
    if(std::filesystem::exists(test_path)) {
        ret.push_back(test_path.string());
    } else {
        SPDLOG_DEBUG("无效的测试训练集：{}", test_path.string());
    }
    return ret;
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
    const std::function<void(const std::string&, std::vector<torch::Tensor>&, std::vector<torch::Tensor>&, const torch::DeviceType&)> transform
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

void lifuren::dataset::image::mask(cv::Mat& mask, const cv::Mat& prev, const cv::Mat& next) {
    cv::Mat diff = next - prev;
    cv::resize(diff, diff, cv::Size(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT), 0, 0, cv::INTER_AREA);
    cv::threshold(diff, diff, LFR_VIDEO_BLACK_MEAN, 255, cv::THRESH_BINARY);
    std::vector<cv::Mat> channels;
    cv::split(diff, channels);
    mask = channels[0] + channels[1] + channels[2];
    cv::resize(mask, mask, cv::Size(LFR_VIDEO_MAT_WIDTH, LFR_VIDEO_MAT_HEIGHT), 0, 0, cv::INTER_AREA);
    // 不二值化更加丰富
    // cv::threshold(mask, mask, LFR_VIDEO_BLACK_MEAN, 255, cv::THRESH_BINARY);
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
    return torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({2, 0, 1}).to(torch::kFloat32).div(255.0).mul(2.0).sub(1.0).contiguous();
}

void lifuren::dataset::image::tensor_to_mat(cv::Mat& image, const torch::Tensor& tensor) {
    if(image.empty()) {
        return;
    }
    if(tensor.dim() == 3) {
        auto image_tensor = tensor.permute({1, 2, 0}).add(1.0).mul(255.0).div(2.0).to(torch::kByte).contiguous();
        std::memcpy(image.data, reinterpret_cast<char*>(image_tensor.data_ptr()), image.total() * image.elemSize());
    } else {
        int i = 0;
        auto image_tensor  = tensor.permute({0, 2, 3, 1}).add(1.0).mul(255.0).div(2.0).to(torch::kByte).contiguous();
        auto images_tensor = image_tensor.chunk(image_tensor.size(0), 0);
        for(auto iter = images_tensor.begin(); iter != images_tensor.end(); ++iter) {
            cv::Mat copy(image_tensor.size(1), image_tensor.size(2), CV_8UC3);
            std::memcpy(copy.data, reinterpret_cast<char*>(iter->data_ptr()), copy.total() * copy.elemSize());
            int x = i % (image.cols / copy.cols) * copy.cols;
            int y = i / (image.cols / copy.cols) * copy.rows;
            copy.copyTo(image(cv::Rect(x, y, copy.cols, copy.rows)));
            ++i;
        }
    }
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
}

lifuren::dataset::RndDatasetLoader lifuren::dataset::image::loadWudaoziDatasetLoader(const int width, const int height, const size_t batch_size, const std::string& path) {
    size_t frame_count = 0;
    size_t video_count = 0;
    std::vector<torch::Tensor> time_step(LFR_VIDEO_FRAME_MAX);
    for(int i = 0; i < LFR_VIDEO_FRAME_MAX; ++i) {
        time_step[i] = torch::tensor({ i }).to(lifuren::get_device());
    }
    auto dataset = lifuren::dataset::Dataset(
        batch_size,
        path,
        { ".mp4" },
        [width, height, &time_step, &frame_count, &video_count] (
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
            cv::Mat old_frame; // 上次视频帧
            cv::Mat src_frame; // 当前视频帧
            std::vector<torch::Tensor> batch;
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
                if(mean <= LFR_VIDEO_BLACK_MEAN) {
                    // 跳过黑屏或者暗屏
                    continue;
                }
                if(!old_frame.empty()) {
                    cv::absdiff(src_frame, old_frame, diff);
                    mean = cv::mean(diff)[0];
                    if(mean == 0) {
                        // 没有变化
                        continue;
                    } else if(mean > LFR_VIDEO_DIFF || batch.size() >= LFR_VIDEO_FRAME_MAX) {
                        if(batch.size() >= LFR_VIDEO_FRAME_MIN) {
                            SPDLOG_DEBUG("加载视频片段：{}", batch.size());
                            frame += batch.size();
                            auto feature = torch::stack(batch, 0).clone();
                            // auto feature = torch::stack(batch, 0).clone().to(device);
                            for(size_t i = 0; i < batch.size() - 1; ++i) {
                                features.push_back(feature.slice(0, i, i + 2));
                                // features.push_back(feature.slice(0, i, i + 2).to(lifuren::get_device()));
                                labels  .push_back(time_step[i]); // TODO: 方向
                            }
                        } else {
                            SPDLOG_DEBUG("丢弃视频片段：{}", batch.size());
                        }
                        batch.clear();
                    } else {
                        batch.push_back(lifuren::dataset::image::mat_to_tensor(old_frame));
                    }
                }
                old_frame = src_frame;
            }
            if(batch.size() >= LFR_VIDEO_FRAME_MIN) {
                SPDLOG_DEBUG("加载视频片段：{}", batch.size());
                frame += batch.size();
                auto feature = torch::stack(batch, 0).clone();
                // auto feature = torch::stack(batch, 0).clone().to(device);
                for(size_t i = 0; i < batch.size() - 1; ++i) {
                    features.push_back(feature.slice(0, i, i + 2));
                    // features.push_back(feature.slice(0, i, i + 2).to(lifuren::get_device()));
                    labels  .push_back(time_step[i]); // 方向
                }
            } else {
                SPDLOG_DEBUG("丢弃视频片段：{}", batch.size());
            }
            batch.clear();
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
