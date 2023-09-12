#include "../header/Mark.hpp"

std::string lifuren::Mark::toJSON() {
    nlohmann::json mark   = nlohmann::json::object();
    nlohmann::json labels = nlohmann::json::array();
    for(std::vector<std::string>::iterator iterator = this->labels.begin(); iterator != this->labels.end(); iterator++) {
        labels.push_back(*iterator);
    }
    mark["file"] = this->file;
    mark["hash"] = this->hash;
    mark["labels"] = labels;
    return mark.dump();
}

std::string lifuren::MarkText::toJSON() {
    nlohmann::json mark   = nlohmann::json::object();
    nlohmann::json labels = nlohmann::json::array();
    for(std::vector<std::string>::iterator iterator = this->labels.begin(); iterator != this->labels.end(); iterator++) {
        labels.push_back(*iterator);
    }
    mark["file"] = this->file;
    mark["hash"] = this->hash;
    mark["name"] = this->name;
    mark["text"] = this->text;
    mark["labels"] = labels;
    return mark.dump();
}