/**
 * 绘画终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_PAINT_CLIENT_HPP
#define LFR_HEADER_CV_PAINT_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

struct PaintParams {

    std::string model;  // 模型
    std::string image;  // 图片文件
    std::string output; // 输出文件
    
};

using PaintModelClient = ModelClient<lifuren::config::ModelParams, PaintParams, std::string>;

template<typename M>
using PaintModelImplClient = ModelImplClient<lifuren::config::ModelParams, PaintParams, std::string, M>;

extern std::unique_ptr<lifuren::PaintModelClient> getPaintClient(const std::string& client);


/**
 * 绘画终端
 */
template<typename M>
class PaintClient : public PaintModelImplClient<M> {

public:
    PaintClient();
    virtual ~PaintClient();

public:
public:
    std::string pred(const PaintParams& input) override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_PAINT_CLIENT_HPP
