#include "../header/Setting.hpp"

lifuren::Setting::Setting() {
    this->activation = lifuren::Activation::RELU;
    this->learningRate = 0.0;
    this->regularization = lifuren::Regularization::NONE;
    this->regularizationRate = 0.0;
}

lifuren::Setting::~Setting() {
}

lifuren::Setting::Setting(const std::string& json) {
    *this = nlohmann::json::parse(json);
}

std::string lifuren::Setting::toJSON() {
    const nlohmann::json json = *this;
    return json.dump();
}
