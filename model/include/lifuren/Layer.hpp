/**
 * Layer
 * 
 * @author acgist
 * 
 * 参考资料：
 * 
 * https://github.com/ggerganov/ggml/blob/master/examples/mnist/mnist-common.h
 * https://github.com/ggerganov/ggml/blob/master/examples/mnist/mnist-common.cpp
 * https://github.com/ggerganov/ggml/blob/master/examples/mnist/mnist-train.cpp
 * 
 * https://pytorch.org/docs/stable/nn.html
 * 
 * https://github.com/pytorch/pytorch/blob/main/torch/csrc/
 * https://github.com/pytorch/pytorch/tree/main/torch/csrc/api/include/torch/nn/functional/
 * 
 * https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/
 * https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp
 */
#ifndef LFR_HEADER_MODEL_LAYER_HPP
#define LFR_HEADER_MODEL_LAYER_HPP

#include <map>
#include <string>
#include <memory>

#include "ggml.h"

namespace lifuren {

namespace loss {

/**
 * L1Loss
 * https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
 * 
 * 回归任务
 * 
 * @param ctx    计算上下文
 * @param source 原始数据
 * @param target 目标数据
 * 
 * @return 损失函数
 */
inline ggml_tensor* l1Loss(ggml_context* ctx, ggml_tensor* source, ggml_tensor* target) {
    return ggml_mean(ctx, ggml_transpose(ctx, ggml_abs(ctx, ggml_sub(ctx, source, target))));
}

/**
 * BCELoss
 * https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
 * 
 * 二分类任务
 * 
 * @param ctx    计算上下文
 * @param source 原始数据
 * @param target 目标数据
 * 
 * @return 损失函数
 */
inline ggml_tensor* bceLoss(ggml_context* ctx, ggml_tensor* source, ggml_tensor* target) {
    ggml_tensor* s1 = ggml_div(ctx, source, source);
    ggml_tensor* t1 = ggml_div(ctx, target, target);
    return ggml_mean(ctx, ggml_transpose(ctx, ggml_neg(ctx, ggml_add(
        ctx,
        ggml_mul(ctx, target, ggml_log(ctx, source)),
        ggml_mul(ctx, ggml_add(ctx, t1, ggml_neg(ctx, target)), ggml_log(ctx, ggml_add(ctx, s1, ggml_neg(ctx, source))))
    ))));
}

/**
 * MSELoss
 * https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
 * 
 * 回归任务
 * 
 * @param ctx    计算上下文
 * @param source 原始数据
 * @param target 目标数据
 * 
 * @return 损失函数
 */
inline ggml_tensor* mseLoss(ggml_context* ctx, ggml_tensor* source, ggml_tensor* target) {
    return ggml_mean(ctx, ggml_transpose(ctx, ggml_sqr(ctx, ggml_sub(ctx, source, target))));
}

/**
 * NLLLoss
 * https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
 * 
 * 多分类任务
 * 
 * @param ctx    计算上下文
 * @param source 原始数据
 * @param target 目标数据
 * 
 * @return 损失函数
 */
inline ggml_tensor* nllLoss(ggml_context* ctx, ggml_tensor* source, ggml_tensor* target) {
    // TODO: 实现
    return nullptr;
}

/**
 * SmoothL1Loss
 * https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss
 * 
 * 回归任务
 * 检测任务
 * 
 * @param ctx    计算上下文
 * @param source 原始数据
 * @param target 目标数据
 * 
 * @return 损失函数
 */
inline ggml_tensor* smoothL1Loss(ggml_context* ctx, ggml_tensor* source, ggml_tensor* target) {
    // TODO: 实现
    return nullptr;
}

/**
 * CrossEntropyLoss
 * https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
 * 
 * 分割任务
 * 多分类任务
 * 
 * @param ctx    计算上下文
 * @param source 原始数据
 * @param target 目标数据
 * 
 * @return 损失函数
 */
inline ggml_tensor* crossEntropyLoss(ggml_context* ctx, ggml_tensor* source, ggml_tensor* target) {
    return ggml_cross_entropy_loss(ctx, source, target);
}

}

namespace layer {

/**
 * Layer
 */
class Layer {

protected:
    ggml_context* ctx_weight { nullptr }; // 权重上下文
    ggml_context* ctx_compute{ nullptr }; // 计算上下文
    std::string name;                     // 名称

public:
    Layer(ggml_context* ctx_weight, ggml_context* ctx_compute, const std::string& name = "layer");
    virtual ~Layer();

public:
    virtual std::string  info() const;
    virtual ggml_tensor* forward   (ggml_tensor* input) = 0;
    virtual ggml_tensor* operator()(ggml_tensor* input);
    virtual void defineWeight(      std::map<std::string, ggml_tensor*>& weights) = 0;
    virtual void bindWeight  (const std::map<std::string, ggml_tensor*>& weights) = 0;
    virtual void bindWeight  (const std::map<std::string, ggml_tensor*>& weights, const std::string& key, ggml_tensor** tensor);

};

/**
 * 线性层/全连接层
 */
class Linear : public Layer {

private:
    ggml_tensor* weight{ nullptr }; // 权重
    ggml_tensor* bias  { nullptr }; // 偏置
    size_t in_features { 0 };       // 输入特征大小
    size_t out_features{ 0 };       // 输出特征大小
    bool   bias_       { true };    // 是否添加偏置

public:
    Linear(
        size_t in_features,
        size_t out_features,
        ggml_context* ctx,
        const std::string& name = "linear",
        bool bias = true
    );
    Linear(
        size_t in_features,
        size_t out_features,
        ggml_context* ctx_weight,
        ggml_context* ctx_compute,
        const std::string& name = "linear",
        bool bias = true
    );
    ~Linear();

public:
    using Layer::bindWeight;
    std::string info() const override;
    ggml_tensor* forward(ggml_tensor* input) override;
    void defineWeight(      std::map<std::string, ggml_tensor*>& weights) override;
    void bindWeight  (const std::map<std::string, ggml_tensor*>& weights) override;

};

/**
 * @param in_features  输入特征大小
 * @param out_features 输出特征大小
 * @param ctx_weight   权重上下文
 * @param ctx_compute  计算上下文
 * @param name         名称
 * @param bias         偏置
 * 
 * @return Linear
 */
inline std::unique_ptr<Linear> linear(
    size_t in_features,
    size_t out_features,
    ggml_context* ctx_weight,
    ggml_context* ctx_compute,
    const std::string& name = "linear",
    bool bias = true
) {
    return std::make_unique<Linear>(in_features, out_features, ctx_weight, ctx_compute, name, bias);
}

/**
 * 卷积层
 */
class Conv2d : public Layer {

private:
    ggml_tensor* kernel{ nullptr }; // 卷积核
    ggml_tensor* bias  { nullptr }; // 偏置
    size_t in_channels { 0 };       // 输入通道大小
    size_t out_channels{ 0 };       // 输出通道大小
    size_t kernel_size { 0 };       // 卷积核大小
    size_t stride      { 1 };       // 步长
    size_t padding     { 0 };       // 填充
    size_t dilation    { 1 };       // 间隔
    bool   bias_       { true };    // 是否添加偏置

public:
    Conv2d(
        size_t in_channels,
        size_t out_channels,
        size_t kernel_size,
        ggml_context* ctx,
        const std::string& name = "conv2d",
        size_t stride   = 1,
        size_t padding  = 0,
        size_t dilation = 1,
        bool   bias     = true
    );
    Conv2d(
        size_t in_channels,
        size_t out_channels,
        size_t kernel_size,
        ggml_context* ctx_weight,
        ggml_context* ctx_compute,
        const std::string& name = "conv2d",
        size_t stride   = 1,
        size_t padding  = 0,
        size_t dilation = 1,
        bool   bias     = true
    );
    ~Conv2d();

public:
    using Layer::bindWeight;
    std::string info() const override;
    ggml_tensor* forward(ggml_tensor* input) override;
    void defineWeight(      std::map<std::string, ggml_tensor*>& weights) override;
    void bindWeight  (const std::map<std::string, ggml_tensor*>& weights) override;

};

/**
 * @param in_channels  输入通道大小
 * @param out_channels 输出通道大小
 * @param kernel_size  卷积核大小
 * @param ctx_weight   权重上下文
 * @param ctx_compute  计算上下文
 * @param name         名称
 * @param stride       步长
 * @param padding      填充
 * @param dilation     间隔
 * @param bias         偏置
 * 
 * @return Conv2d
 */
inline std::unique_ptr<Conv2d> conv2d(
    size_t in_channels,
    size_t out_channels,
    size_t kernel_size,
    ggml_context* ctx_weight,
    ggml_context* ctx_compute,
    const std::string& name = "conv2d",
    size_t stride   = 1,
    size_t padding  = 0,
    size_t dilation = 1,
    bool   bias     = true
) {
    return std::make_unique<Conv2d>(in_channels, out_channels, kernel_size, ctx_weight, ctx_compute, name, stride, padding, dilation, bias);
}

/**
 * @param input_size  ?
 * @param hidden_size ?
 * @param num_layer   ?
 * @param bias        ?
 * @param dropout     ?
 * 
 * @return GRU
 */
inline void gru(
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layer = 1,
    bool    bias      = true,
    double  dropout   = 0.0
) {
    // TODO
}

/**
 * @param input_size  ?
 * @param hidden_size ?
 * @param num_layer   ?
 * @param bias        ?
 * @param dropout     ?
 * 
 * @return LSTM
 */
inline void lstm(
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layer = 1,
    bool    bias      = true,
    double  dropout   = 0.0
) {
    // TODO
}

/**
 * @param kernel_size 池化核大小
 * @param input       输入张量
 * @param ctx_compute 计算上下文
 * @param stride      步长
 * @param padding     填充
 * 
 * @return 平均池化
 */
inline ggml_tensor* avgPool2d(
    int64_t kernel_size,
    ggml_tensor * input,
    ggml_context* ctx_compute,
    int64_t stride  = 1,
    int64_t padding = 0
) {
    return ggml_pool_2d(
        ctx_compute, input, GGML_OP_POOL_AVG,
        kernel_size, kernel_size,
        stride,      stride,
        padding,     padding
    );
}

/**
 * @param kernel_size 池化核大小
 * @param input       输入张量
 * @param ctx_compute 计算上下文
 * @param stride      步长
 * @param padding     填充
 * 
 * @return 最大池化
 */
inline ggml_tensor* maxPool2d(
    int64_t kernel_size,
    ggml_tensor * input,
    ggml_context* ctx_compute,
    int64_t stride  = 1,
    int64_t padding = 0
) {
    return ggml_pool_2d(
        ctx_compute, input, GGML_OP_POOL_MAX,
        kernel_size, kernel_size,
        stride,      stride,
        padding,     padding
    );
}

inline void flatten() {
    // TODO
}

/**
 * 丢弃层
 */
inline void dropout1d() {
    // TODO
}

/**
 * 丢弃层
 */
inline void dropout2d() {
    // TODO
}

/**
 * 组归一化
 */
inline void groupNorm() {
    // TODO
}

/**
 * 批标准化层
 */
inline void batchNorm1d() {
    // TODO
}

/**
 * 批标准化层
 */
inline void batchNorm2d() {
    // TODO
}

} // END layer
} // END lifuren

#endif // LFR_HEADER_MODEL_LAYER_HPP
