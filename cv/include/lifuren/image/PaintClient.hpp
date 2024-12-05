/**
 * 绘画终端
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CV_PAINT_CLIENT_HPP
#define LFR_HEADER_CV_PAINT_CLIENT_HPP

#include "lifuren/Client.hpp"

namespace lifuren {

struct PaintConfigOptions {

    std::string model;         // 模型
    std::string train_path {}; // 训练数据集路径
    std::string val_path   {}; // 验证数据集路径
    std::string test_path  {}; // 测试数据集路径

};

struct PaintOptions {

    std::string image;  // 图片文件
    std::string output; // 输出文件
    
};

using PaintModelClient = ModelClient<PaintConfigOptions, PaintOptions, std::string>;

template<typename M>
using PaintModelImplClient = ModelImplClient<PaintConfigOptions, PaintOptions, std::string, M>;

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
    std::string pred(const PaintOptions& input) override;

};

} // END OF lifuren

#endif // END OF LFR_HEADER_CV_PAINT_CLIENT_HPP
