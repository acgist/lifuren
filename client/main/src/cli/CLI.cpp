#include "lifuren/CLI.hpp"

#include <string>
#include <vector>
#include <iostream>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Message.hpp"
#include "lifuren/Wudaozi.hpp"

static void pred (const std::vector<std::string>&); // 预测
static void train(const std::vector<std::string>&); // 训练
static void help (); // 帮助

static void messageCallback(const char*); // 消息回调

bool lifuren::cli(const int argc, const char* const argv[]) {
    if(argc <= 1) {
        return false;
    }
    lifuren::message::register_message_callback(messageCallback);
    std::vector<std::string> args;
    for(int i = 0; i < argc; ++i) {
        SPDLOG_DEBUG("命令参数：{} - {}", i, argv[i]);
        if(i < 2) {
            continue;
        }
        args.push_back(argv[i]);
    }
    const char* const command = argv[1];
    if(std::strcmp(command, "pred") == 0) {
        ::pred(args);
    } else if(std::strcmp(command, "train") == 0) {
        ::train(args);
    } else {
        ::help();
    }
    lifuren::message::unregister_message_callback();
    return true;
}

static void pred(const std::vector<std::string>& args) {
    if(args.size() < 2) {
        SPDLOG_WARN("缺少参数");
        ::help();
        return;
    }
    auto client = lifuren::get_wudaozi_client();
    if(!client) {
        SPDLOG_WARN("无效模型");
        return;
    }
    const std::string& model_file = args[0];
    const std::string& image_file = args[1];
    client->load(model_file);
    const auto [success, output_file] = client->pred(image_file);
    if(success) {
        SPDLOG_INFO("生成完成：{}", output_file);
    } else {
        SPDLOG_WARN("生成失败：{}", output_file);
    }
}

static void train(const std::vector<std::string>& args) {
    if(args.size() < 2) {
        SPDLOG_WARN("缺少参数");
        ::help();
        return;
    }
    auto client = lifuren::get_wudaozi_client();
    if(!client) {
        SPDLOG_WARN("无效模型");
        return;
    }
    const std::string& model_path = args[2];
    const std::string& dataset    = args[3];
    lifuren::config::ModelParams params{
        .model_name = "wudaozi",
        .model_path = model_path,
        .train_path = lifuren::file::join({dataset, lifuren::config::DATASET_TRAIN}).string(),
        .val_path   = lifuren::file::join({dataset, lifuren::config::DATASET_VAL  }).string(),
        .test_path  = lifuren::file::join({dataset, lifuren::config::DATASET_TEST }).string(),
    };
    client->trainValAndTest(params);
    client->save(lifuren::file::join({model_path, "wudaozi.pt"}).string());
}

static void help() {
    std::cout << R"(
./lifuren[.exe] 命令 [参数...]
./lifuren[.exe] [pred|train] [model_file|model_path] [image_file|dataset]
./lifuren[.exe] [?|help]
)" << std::endl;
}

static void messageCallback(const char* message) {
    #if defined(_DEBUG) || !defined(NDEBUG)
    // 测试时控制台日志已经打开
    #else
    std::cout << message << std::endl;
    #endif
}
