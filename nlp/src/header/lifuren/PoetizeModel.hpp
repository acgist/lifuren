/**
 * 诗词模型
 */
#ifndef LFR_HEADER_NLP_POETIZE_MODEL_HPP
#define LFR_HEADER_NLP_POETIZE_MODEL_HPP

#include "lifuren/Model.hpp"

namespace lifuren {

/**
 * 诗词模型
 */
class PoetizeModel {
};

/**
 * 诗佛模型
 */
class ShifoRNNModel : public PoetizeModel {
};

/**
 * 诗魔模型
 */
class ShimoRNNModel : public PoetizeModel {
};

/**
 * 诗鬼模型
 */
class ShiguiRNNModel : public PoetizeModel {
};

/**
 * 诗仙模型
 */
class ShixianRNNModel : public PoetizeModel {
};

/**
 * 诗圣模型
 */
class ShishengRNNModel : public PoetizeModel {
};

/**
 * 李杜模型
 */
class LiduRNNModel : public PoetizeModel {
};

/**
 * 苏辛模型
 */
class SuxinRNNModel : public PoetizeModel {
};

/**
 * 婉约模型
 */
class WanyueRNNModel : public PoetizeModel {
};

}

#endif // END OF LFR_HEADER_NLP_POETIZE_MODEL_HPP