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

};

};

template<>
std::tuple<bool, std::string> lifuren::image::ImageClient<lifuren::image::WudaoziModel>::pred(const std::string& input) {
    if(!this->model) {
        return { false, {} };
    }
    // TODO: 图片列表 排序 切割
    return {};
}

std::unique_ptr<lifuren::image::ImageModelClient> lifuren::image::getImageClient(const std::string& model) {
    if(model == "wudaozi") {
        return std::make_unique<lifuren::image::ImageClient<WudaoziModel>>();
    } else {
        return nullptr;
    }
}
