/**
 * 绘画终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_PAINT_CLIENT_HPP
#define LFR_HEADER_CV_PAINT_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

struct PaintOptions {

    std::string model;  // 模型
    std::string image;  // 图片
    std::string output; // 输出
    
};

using PaintModelClient = ModelClient<PaintOptions, std::string>;

template<typename M>
using PaintModelImplClient = ModelImplClient<PaintOptions, std::string, M>;

extern std::unique_ptr<lifuren::PaintModelClient> getPaintClient(const std::string& client);


/**
 * 绘画终端
 */
template<typename M>
class PaintClient : public StatefulClient, public PaintModelImplClient<M> {

public:
    PaintClient();
    virtual ~PaintClient();

public:
public:
    std::string pred(const PaintOptions& input) override;
    void        pred(const PaintOptions& input, PaintModelClient::Callback callback) override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_PAINT_CLIENT_HPP
