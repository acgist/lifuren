#include "lifuren/poetry/Poetry.hpp"

#include "lifuren/poetry/PoetryModel.hpp"

template<typename M>
std::tuple<bool, std::string> lifuren::poetry::PoetryClient<M>::pred(const PoetryParams& input) {
    // TODO: 实现
    return {};
};

std::unique_ptr<lifuren::poetry::PoetryModelClient> lifuren::poetry::getPoetryClient(const std::string& client) {
    if(client == lifuren::config::CONFIG_POETRY_LIDU) {
        return std::make_unique<lifuren::poetry::PoetryClient<LiduModel>>();
    } else if(client == lifuren::config::CONFIG_POETRY_SUXIN) {
        return std::make_unique<lifuren::poetry::PoetryClient<SuxinModel>>();
    } else {
        return nullptr;
    }
}
