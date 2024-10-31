/**
 * 绘画终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_PAINT_CLIENT_HPP
#define LFR_HEADER_CV_PAINT_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

/**
 * 绘画终端
 */
class PaintClient : public StatefulClient {

public:

struct PaintOptions {

    std::string image;
    std::string model;
    std::string output;
    
};

public:
/**
 * 绘画回调
 * 
 * @param finish  是否完成
 * @param percent 进度
 * @param message 没有完成=提示内容、任务完成=图片路径
 * 
 * @return 是否结束
 */
using PaintCallback = std::function<bool(bool finish, float percent, const std::string& message)>;

protected:
    PaintCallback callback{ nullptr };

public:
    static std::unique_ptr<lifuren::PaintClient> getClient(const std::string& client);

public:
    PaintClient(PaintCallback callback = nullptr);
    virtual ~PaintClient();

public:
    /**
     * @param options  提示内容
     * @param callback 消息回调
     * 
     * @return 是否成功
     */
    virtual bool paint(const PaintOptions& options, PaintCallback callback = nullptr) = 0;

};

class CycleGANPaintClient : public PaintClient {

public:
    CycleGANPaintClient();
    virtual ~CycleGANPaintClient();

public:
    bool paint(const PaintOptions& options, PaintClient::PaintCallback callback = nullptr) override;
    
};

class StyleGANPaintClient : public PaintClient {

public:
    StyleGANPaintClient();
    virtual ~StyleGANPaintClient();

public:
    bool paint(const PaintOptions& options, PaintClient::PaintCallback callback = nullptr) override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_PAINT_CLIENT_HPP
