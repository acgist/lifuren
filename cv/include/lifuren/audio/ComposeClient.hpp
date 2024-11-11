/**
 * 作曲终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_COMPOSE_CLIENT_HPP
#define LFR_HEADER_CV_COMPOSE_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

/**
 * 作曲终端
 */
class ComposeClient : public StatefulClient {

public:

struct ComposeOptions {

    std::string model;
    std::string audio;
    std::string output;
    
};

public:
/**
 * 作曲回调
 * 
 * @param finish  是否完成
 * @param percent 进度
 * @param message 没有完成=提示内容、任务完成=图片路径
 * 
 * @return 是否结束
 */
using ComposeCallback = std::function<bool(bool finish, float percent, const std::string& message)>;

protected:
    ComposeCallback callback{ nullptr };

public:
    static std::unique_ptr<lifuren::ComposeClient> getClient(const std::string& client);

public:
    ComposeClient(ComposeCallback callback = nullptr);
    virtual ~ComposeClient();

public:
    /**
     * @param options  提示内容
     * @param callback 消息回调
     * 
     * @return 是否成功
     */
    virtual bool paint(const ComposeOptions& options, ComposeCallback callback = nullptr) = 0;

};

class ShikuangComposeClient : public ComposeClient {

public:
    ShikuangComposeClient();
    virtual ~ShikuangComposeClient();

public:
    bool paint(const ComposeOptions& options, ComposeClient::ComposeCallback callback = nullptr) override;
    
};

class LiguinianComposeClient : public ComposeClient {

public:
    LiguinianComposeClient();
    virtual ~LiguinianComposeClient();

public:
    bool paint(const ComposeOptions& options, ComposeClient::ComposeCallback callback = nullptr) override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_COMPOSE_CLIENT_HPP
