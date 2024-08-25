#include "lifuren/Client.hpp"

#include "spdlog/spdlog.h"

#if _WIN32
#ifndef popen
#define popen _popen
#endif
#ifndef pclose
#define pclose _pclose
#endif
#endif

lifuren::CommandClient::CommandClient(const std::string& command, std::function<void(bool, const std::string&)> callback) : command(command), callback(callback) {
}

lifuren::CommandClient::~CommandClient() {
    if(this->pipe) {
        pclose(this->pipe);
    }
}

const int& lifuren::CommandClient::execute() {
    if(this->pipe) {
        SPDLOG_WARN("命令已经执行：{}", this->command);
        return this->code;
    }
    if(this->command.empty()) {
        SPDLOG_WARN("执行命令终端：命令为空");
        return this->code;
    }
    this->pipe = popen(this->command.c_str(), "r");
    if(!this->pipe) {
        SPDLOG_WARN("执行命令终端失败：{}", this->command);
        return this->code;
    }
    char buffer[128];
    while(fgets(buffer, 128, this->pipe)) {
        if(this->callback) {
            this->callback(false, buffer);
        }
        this->result += buffer;
    }
    this->callback(true, "");
    this->code = fclose(this->pipe);
    return this->code;
}

void lifuren::CommandClient::shutdown() const {
    if(this->pipe) {
        SPDLOG_DEBUG("关闭命令终端：{}", this->command);
        pclose(pipe);
    } else {
        SPDLOG_DEBUG("关闭命令无效终端：{}", this->command);
    }
}

const int& lifuren::CommandClient::getCode() const {
    return this->code;
}

const std::string& lifuren::CommandClient::getResult() const {
    return this->result;
}
