/**
 * 导演终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_ACT_CLIENT_HPP
#define LFR_HEADER_CV_ACT_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

/**
 * 导演终端
 */
class ActClient : public StatefulClient {

public:

struct ActOptions {

    std::string video;
    std::string model;
    std::string output;
    
};

public:
/**
 * 导演回调
 * 
 * @param finish  是否完成
 * @param percent 进度
 * @param message 没有完成=提示内容、任务完成=图片路径
 * 
 * @return 是否结束
 */
using ActCallback = std::function<bool(bool finish, float percent, const std::string& message)>;

protected:
    ActCallback callback{ nullptr };

public:
    static std::unique_ptr<lifuren::ActClient> getClient(const std::string& client);

public:
    ActClient(ActCallback callback = nullptr);
    virtual ~ActClient();

public:
    /**
     * @param options  提示内容
     * @param callback 消息回调
     * 
     * @return 是否成功
     */
    virtual bool paint(const ActOptions& options, ActCallback callback = nullptr) = 0;

};

class GuanhanqinActClient : public ActClient {

public:
    GuanhanqinActClient();
    virtual ~GuanhanqinActClient();

public:
    bool paint(const ActOptions& options, ActClient::ActCallback callback = nullptr) override;
    
};

class TangxianzuActClient : public ActClient {

public:
    TangxianzuActClient();
    virtual ~TangxianzuActClient();

public:
    bool paint(const ActOptions& options, ActClient::ActCallback callback = nullptr) override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_ACT_CLIENT_HPP
