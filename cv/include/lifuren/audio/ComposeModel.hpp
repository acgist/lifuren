/**
 * 作曲模型
 */
#ifndef LFR_HEADER_CV_COMPOSE_DATASET_HPP
#define LFR_HEADER_CV_COMPOSE_DATASET_HPP

#include "lifuren/Model.hpp"
#include "lifuren/audio/AudioDataset.hpp"

namespace lifuren {

class ShikuangModuleImpl : public torch::nn::Module {

    // 卷积->卷积->GRU GRU 还原->还原

public:
    torch::Tensor forward(torch::Tensor input);

};

TORCH_MODULE(ShikuangModule);

// class ShikuangModel : public lifuren::Model<

// > {
//     // TODO: 实现
// };

class LiguinianModel {
    // TODO: 实现
};

}

#endif // END OF LFR_HEADER_CV_COMPOSE_DATASET_HPP
