/**
 * Layer工具
 * 
 * @author acgist
 * 
 * https://github.com/pytorch/pytorch/blob/main/torch/csrc/
 * https://github.com/pytorch/pytorch/tree/main/torch/csrc/api/include/torch/nn/functional/
 */
#ifndef LFR_HEADER_MODEL_LAYERS_HPP
#define LFR_HEADER_MODEL_LAYERS_HPP

#include <map>
#include <vector>
#include <string>

struct ggml_tensor;
struct ggml_context;

namespace lifuren {
namespace layers  {

class Layer {

protected:
    std::string name;
    ggml_context* ctx_weight { nullptr };
    ggml_context* ctx_compute{ nullptr };

public:
    Layer(ggml_context* ctx_weight, ggml_context* ctx_compute, const std::string& name = "");
    virtual ~Layer();

public:
    virtual std::string info();
    virtual ggml_tensor* forward   (ggml_tensor* input) = 0;
    virtual ggml_tensor* operator()(ggml_tensor* input);
    virtual void defineWeight(std::map<std::string, ggml_tensor*>& weights) = 0;
    virtual void bindWeight  (std::map<std::string, ggml_tensor*>& weights) = 0;
    virtual void bindWeight  (std::map<std::string, ggml_tensor*>& weights, const std::string& key, ggml_tensor** tensor);

};

class Linear : public Layer {

private:
    ggml_tensor* weight{ nullptr };
    ggml_tensor* bias  { nullptr };
    size_t in_features { 0 };
    size_t out_features{ 0 };

public:
    Linear(size_t in_features, size_t out_features, ggml_context* ctx, const std::string& name = "linear");
    Linear(size_t in_features, size_t out_features, ggml_context* ctx_weight, ggml_context* ctx_compute, const std::string& name = "linear");
    ~Linear();

public:
    using Layer::bindWeight;
    std::string info() override;
    ggml_tensor* forward(ggml_tensor* input) override;
    void defineWeight(std::map<std::string, ggml_tensor*>& weights) override;
    void bindWeight  (std::map<std::string, ggml_tensor*>& weights) override;

};

/**
 * @param in_features  输入特征大小
 * @param out_features 输出特征大小
 * 
 * @return Linear
 */
inline void linear(int64_t in_features, int64_t out_features) {
    // TODO
}

/**
 * @param in_channels  输入通道大小
 * @param out_channels 输出通道大小
 * @param kernel_size  卷积核大小
 * @param stride       步长
 * @param padding      填充
 * @param dilation     间隔
 * @param bias         偏置
 * 
 * @return Conv2d
 */
inline void conv2d(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride   = 1,
    int64_t padding  = 0,
    int64_t dilation = 1,
    bool    bias     = true
) {
    // TODO
}

/**
 * @param kernel_size 卷积核大小
 * @param stride      步长
 * @param padding     填充
 * @param dilation    间隔
 * 
 * @return MaxPool2d
 */
inline void maxPool2d(
    int64_t kernel_size,
    int64_t stride   = -1,
    int64_t padding  = 0,
    int64_t dilation = 1
) {
    // TODO
}

/**
 * @param kernel_size 卷积核大小
 * @param stride      步长
 * @param padding     填充
 * 
 * @return MaxPool2d
 */
inline void avgPool2d(
    int64_t kernel_size,
    int64_t stride  = -1,
    int64_t padding = 0
) {
    // TODO
}

/**
 * @param input_size  ?
 * @param hidden_size ?
 * @param num_layers  ?
 * @param bias        ?
 * @param dropout     ?
 * 
 * @return GRU
 */
inline void gru(
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layers = 1,
    bool    bias       = true,
    double  dropout    = 0.0
) {
    // TODO
}

/**
 * @param input_size  ?
 * @param hidden_size ?
 * @param num_layers  ?
 * @param bias        ?
 * @param dropout     ?
 * 
 * @return LSTM
 */
inline void lstm(
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layers = 1,
    bool    bias       = true,
    double  dropout    = 0.0
) {
    // TODO
}

} // END layers
} // END lifuren

#endif // LFR_HEADER_MODEL_LAYERS_HPP
