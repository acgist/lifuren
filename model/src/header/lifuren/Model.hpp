/**
 * 模型
 * 
 * @author acgist
 * 
 * https://github.com/ggerganov/ggml/blob/master/examples/mnist/mnist-common.h
 * https://github.com/ggerganov/ggml/blob/master/examples/mnist/mnist-common.cpp
 * https://github.com/ggerganov/ggml/blob/master/examples/mnist/mnist-train.cpp
 * 
 * TODO:
 * 1. datas->features
 */
#ifndef LFR_HEADER_MODEL_MODEL_HPP
#define LFR_HEADER_MODEL_MODEL_HPP

#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;
struct ggml_context;
struct ggml_opt_context;

namespace lifuren {

namespace datasets {
    class Dataset;
}

/**
 * 李夫人模型
 * 
 * @author acgist
 */
class Model {

public:

/**
 * 初始方式
 */
enum class InitType {

    ZERO,
    RAND,
    VALUE,

};

struct OptimizerParams {

    size_t n_iter = 8;

};

struct ModelParams {

    // 学习率
    float lr = 0.001F;
    // 批量大小
    size_t batch_size  = 100;
    // 训练次数
    size_t epoch_count = 128;
    // 线程数量
    size_t thread_size = std::thread::hardware_concurrency();
    // 分类模型
    bool classify = false;
    // 分类数量
    size_t size_classify = 0LL;
    // 计算图的大小
    size_t size_cgraph  = 16LL  * 1024;
    // 权重大小
    size_t size_weight  = 128LL * 1024 * 1024;
    // 计算大小
    size_t size_compute = 256LL * 1024 * 1024;
    // 优化函数参数
    OptimizerParams optimizerParams;

};

protected:
    // 模型参数
    ModelParams params{};
    // 权重上下文
    void        * buf_weight { nullptr };
    ggml_context* ctx_weight { nullptr };
    // 计算上下文
    void        * buf_compute{ nullptr };
    ggml_context* ctx_compute{ nullptr };
    // 损失函数
    ggml_tensor* loss  { nullptr };
    // 目标函数
    ggml_tensor* logits{ nullptr };
    // 输入数据
    ggml_tensor* datas { nullptr };
    // 输入标签
    ggml_tensor* labels{ nullptr };
    // 预测结果
    ggml_tensor* preds { nullptr };
    // 计算图
    ggml_cgraph* train_gf{ nullptr };
    ggml_cgraph* train_gb{ nullptr };
    ggml_cgraph* val_gf  { nullptr };
    ggml_cgraph* test_gf { nullptr };
    ggml_cgraph* eval_gf { nullptr };
    std::map<std::string, ggml_tensor*> weights{};

public:
    // 训练数据集
    std::unique_ptr<lifuren::datasets::Dataset> trainDataset{ nullptr };
    // 验证数据集
    std::unique_ptr<lifuren::datasets::Dataset> valDataset  { nullptr };
    // 测试数据集
    std::unique_ptr<lifuren::datasets::Dataset> testDataset { nullptr };

public:
    Model(ModelParams params);
    virtual ~Model();

protected:
    // 加载上下文
    void initContext();

public:
    // 训练模型保存加载
    virtual bool   save(const std::string& path = "./", const std::string& filename = "lifuren.gguf");
    virtual Model& load(const std::string& path = "./", const std::string& filename = "lifuren.gguf");
    // 预测模型保存加载
    virtual bool   saveEval(const std::string& path = "./", const std::string& filename = "lifuren.ggml");
    virtual Model& loadEval(const std::string& path = "./", const std::string& filename = "lifuren.ggml");
    // 定义模型
    virtual Model& define(InitType type = InitType::RAND, float mean = 0.0F, float sigma = 0.001F, float value = 0.0F);
    // 定义权重
    virtual Model& defineWeight() = 0;
    // 初始化权重
    virtual Model& initWeight(InitType type = InitType::RAND, float mean = 0.0F, float sigma = 0.001F, float value = 0.0F);
    // 绑定权重
    virtual Model& bindWeight() = 0;
    // 初始化输入
    virtual Model&       defineInput();
    virtual ggml_tensor* buildDatas()  = 0;
    virtual ggml_tensor* buildLabels() = 0;
    // 定义计算逻辑
    virtual Model&       defineLogits();
    virtual ggml_tensor* buildLogits() = 0;
    // 定义损失函数
    virtual Model&       defineLoss();
    virtual ggml_tensor* buildLoss()   = 0;
    // 定义计算图
    virtual Model& defineCgraph();
    // 打印模型
    virtual Model& print();
    virtual Model& print(const char* name, const ggml_cgraph* cgraph, std::string& message);
    virtual Model& print(const char* from, const ggml_tensor* tensor, std::string& message);
    // 训练模型
    virtual void train(size_t epoch, ggml_opt_context* opt_ctx);
    // 验证模型
    virtual void val(size_t epoch);
    // 测试模型
    virtual void test();
    // 模型预测
    virtual float* eval(const float* input, float* output, size_t size_data);
    // 模型预测
    virtual std::vector<size_t> evalClassify(const float* input, size_t size_data);
    // 训练验证
    virtual void trainAndVal();
    // 优化函数
    virtual void buildOptimizer(ggml_opt_context* opt_ctx);
    // 正确数量
    virtual size_t batchAccu(const size_t& size);

};

}

#endif // LFR_HEADER_MODEL_MODEL_HPP
