#include "lifuren/Exception.hpp"

lifuren::Exception::Exception(const std::string& code, const std::string& message) : code(code), message(message){
}

lifuren::Exception::~Exception() {
}

void lifuren::Exception::throwException(const std::string& code, const std::string& message) {
    throw lifuren::Exception(code, message);
}

void lifuren::Exception::trueThrow(bool ret, const std::string& code, const std::string& message) {
    if(ret) {
        throw lifuren::Exception(code, message);
    } else {
        // 忽略错误
    }
}

void lifuren::Exception::falseThrow(bool ret, const std::string& code, const std::string& message) {
    if(ret) {
        // 忽略正确
    } else {
        throw lifuren::Exception(code, message);
    }
}

const char* lifuren::Exception::what() const noexcept {
    return this->message.c_str();
}
