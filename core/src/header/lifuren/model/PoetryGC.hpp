/**
 * 诗词内容生成
 * 
 * CycleGAN
 * Transformer
 * 
 * @author acgist
 */
#ifndef LFR_HEADER_CORE_MODEL_POETRYGC_HPP
#define LFR_HEADER_CORE_MODEL_POETRYGC_HPP

#include <string>

#include "../Model.hpp"
#include "./Poetry.hpp"

namespace lifuren {

/**
 * 诗词生成模型设置
 */
class PoetryGCModelConfig : public lifuren::ModelConfig {
};

/**
 * 诗词生成模型
 */
class PoetryGCModel : public Model<PoetryGCModelConfig, std::string, std::string> {

public:
    PoetryGCModel();
    virtual ~PoetryGCModel();
    /**
     * @param config      基本配置
     * @param modelConfig 模型配置
     */
    PoetryGCModel(const lifuren::Config& config, const lifuren::PoetryGCModelConfig& modelConfig);

public:
    void train() override;
    void val() override;
    void test() override;
    std::string pred(std::string path) override;

};

// #include "../../../header/model/PoetryGC.hpp"

// lifuren::PoetryGCModel::PoetryGCModel() {
// }

// lifuren::PoetryGCModel::~PoetryGCModel() {
// }

// lifuren::PoetryGCModel::PoetryGCModel(const lifuren::Config& config, const lifuren::PoetryGCModelConfig& modelConfig) : Model(config, modelConfig) {
// }

// void lifuren::PoetryGCModel::train() {
// }

// void lifuren::PoetryGCModel::val() {
// }

// void lifuren::PoetryGCModel::test() {
// }

// std::string lifuren::PoetryGCModel::pred(std::string) {
//     return "";
// }


/**
 * 图片转为诗词
 */
class ImageToPoetryModel {
};

/**
 * 标签转为诗词
 */
class LabelToPoetryModel {
};

}

#endif // LFR_HEADER_CORE_MODEL_POETRYGC_HPP
