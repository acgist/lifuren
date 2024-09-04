/**
 * 服务终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CLIENT_PAINT_CLIENT_HPP
#define LFR_HEADER_CLIENT_PAINT_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

/**
 * 绘画终端
 */
class PaintClient : public Client {

public:
enum class Mode {
    TXT2IMG,
    IMG2IMG,
    IMG2VID,
    CONVERT,
};
struct PaintOptions {
    Mode mode = Mode::TXT2IMG;

    std::string image;
    std::string video;
    std::string model;
    std::string prompt;
    std::string output;
    
    size_t seed   = 42;
    size_t count  = 1;
    size_t steps  = 30;
    size_t width  = 512;
    size_t height = 512;

    bool color = true;
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
    PaintClient(PaintCallback callback = nullptr);
    ~PaintClient();

public:
    /**
     * @param options  提示内容
     * @param callback 消息回调
     * 
     * @return 是否成功
     */
    virtual bool paint(const PaintOptions& options, PaintCallback callback = nullptr) = 0;

};

class CycleGANPaintClient {
    // TODO: 实现算法
};

class StyleGANPaintClient {
    // TODO: 实现算法
};

/**
 * StableDiffusionCPP终端
 * 
 * https://github.com/leejet/stable-diffusion.cpp
 */
class StableDiffusionCPPPaintClient : public PaintClient {

private:

public:
    StableDiffusionCPPPaintClient();
    ~StableDiffusionCPPPaintClient();

public:
    bool paint(const PaintOptions& options, PaintClient::PaintCallback callback = nullptr) override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CLIENT_PAINT_CLIENT_HPP
