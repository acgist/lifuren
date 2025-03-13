/**
 * Copyright(c) 2024-present acgist. All Rights Reserved.
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * gitee : https://gitee.com/acgist/lifuren
 * github: https://github.com/acgist/lifuren
 * 
 * 图片
 * 
 * @author acgist
 * 
 * @version 1.0.0
 */
#ifndef LFR_HEADER_CORE_IMAGE_HPP
#define LFR_HEADER_CORE_IMAGE_HPP

#include "lifuren/Client.hpp"

namespace lifuren::image {

using ImageModelClient = ModelClient<lifuren::config::ModelParams, std::string, std::string>;

template<typename M>
using ImageModelImplClient = ModelImplClient<lifuren::config::ModelParams, std::string, std::string, M>;

/**
 * 图片终端
 */
template<typename M>
class ImageClient : public ImageModelImplClient<M> {

public:
    std::tuple<bool, std::string> pred(const std::string& input) override;

};

/**
 * @return 视频终端
 */
extern std::unique_ptr<lifuren::image::ImageModelClient> getImageClient(
    const std::string& model // 模型名称
);

}

#endif // END OF LFR_HEADER_CORE_IMAGE_HPP
