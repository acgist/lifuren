#include "lifuren/Score.hpp"

#include <fstream>

#include "lifuren/File.hpp"
#include "lifuren/ScoreModel.hpp"

namespace lifuren::score {

template<typename M>
using ScoreModelClientImpl = ModelClientImpl<lifuren::config::ModelParams, std::string, std::string, M>;

template<typename M>
class ScoreClient : public ScoreModelClientImpl<M> {

public:
    std::tuple<bool, std::string> pred(const std::string& input) override;

};

};

template<>
std::tuple<bool, std::string> lifuren::score::ScoreClient<lifuren::score::MozartModel>::pred(const std::string& input) {
    if(!this->model) {
        return { false, {} };
    }
    // TODO
    return {};
}

std::unique_ptr<lifuren::score::ScoreModelClient> lifuren::score::getScoreClient(const std::string& model) {
    if(model == "mozart") {
        return std::make_unique<lifuren::score::ScoreClient<MozartModel>>();
    } else {
        return nullptr;
    }
}
