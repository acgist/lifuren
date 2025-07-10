#include "lifuren/CLI.hpp"

#include <string>
#include <vector>
#include <iostream>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Message.hpp"
#include "lifuren/Wudaozi.hpp"

static void pred_image(const std::vector<std::string>&); // 预测图片
static void pred_video(const std::vector<std::string>&); // 预测视频
static void train     (const std::vector<std::string>&); // 训练
static void help(); // 帮助

static void message_callback(const char*); // 消息回调

bool lifuren::cli(const int argc, const char* const argv[]) {
    if(argc <= 1) {
        return false;
    }
    lifuren::message::register_message_callback(message_callback);
    std::vector<std::string> args;
    for(int i = 0; i < argc; ++i) {
        SPDLOG_DEBUG("命令参数：{} - {}", i, argv[i]);
        if(i < 2) {
            continue;
        }
        args.push_back(argv[i]);
    }
    const char* const command = argv[1];
    if(std::strcmp(command, "image") == 0) {
        ::pred_image(args);
    } else if(std::strcmp(command, "video") == 0) {
        ::pred_video(args);
    } else if(std::strcmp(command, "train") == 0) {
        ::train(args);
    } else {
        ::help();
    }
    lifuren::message::unregister_message_callback();
    return true;
}

static void pred_image(const std::vector<std::string>& args) {
    if(args.size() < 2) {
        SPDLOG_WARN("缺少参数");
        ::help();
        return;
    }
    auto client = lifuren::get_wudaozi_client();
    const std::string& model_file = args[0];
    const std::string& image_path = args[1];
    if(!client->load(model_file)) {
        SPDLOG_WARN("模型加载失败：{}", model_file);
        return;
    }
    const auto [success, output_file] = client->pred({
        .n    = 1,
        .path = image_path,
        .type = lifuren::WudaoziType::IMAGE
    });
    if(success) {
        SPDLOG_INFO("生成完成：{}", output_file);
    } else {
        SPDLOG_WARN("生成失败：{}", output_file);
    }
}

static void pred_video(const std::vector<std::string>& args) {
    if(args.size() < 2) {
        SPDLOG_WARN("缺少参数");
        ::help();
        return;
    }
    auto client = lifuren::get_wudaozi_client();
    const std::string& model_file = args[0];
    const std::string& image_file = args[1];
    if(!client->load(model_file)) {
        SPDLOG_WARN("模型加载失败：{}", model_file);
        return;
    }
    const auto [success, output_file] = client->pred({
        .file = image_file,
        .type = lifuren::WudaoziType::VIDEO
    });
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
    const std::string& model_path = args[0];
    const std::string& dataset    = args[1];
    const std::string& model_file = args.size() < 3 ? "" : args[2];
    lifuren::config::ModelParams params{
        .model_name = "wudaozi",
        .model_path = model_path,
        .train_path = lifuren::file::join({ dataset, lifuren::config::DATASET_TRAIN }).string(),
        .val_path   = lifuren::file::join({ dataset, lifuren::config::DATASET_VAL   }).string(),
        .test_path  = lifuren::file::join({ dataset, lifuren::config::DATASET_TEST  }).string(),
    };
    if(lifuren::file::exists(model_file) && lifuren::file::is_file(model_file)) {
        client->load(model_file, params);
    }
    client->trainValAndTest(params);
    client->save(lifuren::file::join({ model_path, "wudaozi.pt" }).string());
}

static void help() {
    std::cout << R"(
./lifuren[.exe] 命令 [参数...]
./lifuren[.exe] train model_path dataset [ model_file ]
./lifuren[.exe] image model_file image_path
./lifuren[.exe] video model_file image_file
./lifuren[.exe] [?|help]
)" << std::endl;
}

static void message_callback(const char* message) {
    #if defined(_DEBUG) || !defined(NDEBUG)
    // 测试时控制台日志已经打开
    #else
    std::cout << message << std::endl;
    #endif
}
