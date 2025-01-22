#include "lifuren/CLI.hpp"

#include <string>
#include <vector>
#include <iostream>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Message.hpp"
#include "lifuren/audio/Audio.hpp"
#include "lifuren/video/Video.hpp"
#include "lifuren/poetry/Poetry.hpp"

static void generateAudio  (const std::vector<std::string>&); // 生成音频
static void generateVideo  (const std::vector<std::string>&); // 生成视频
static void generatePoetry (const std::vector<std::string>&); // 生成诗词
static void embedding      (const std::vector<std::string>&); // 数据嵌入
static void embeddingAudio (const std::vector<std::string>&); // 音频嵌入
static void embeddingPepper(const std::vector<std::string>&); // 辣椒嵌入
static void embeddingPoetry(const std::vector<std::string>&); // 诗词嵌入
static void help(); // 帮助
static void messageCallback(bool, const char*); // 消息回调

bool lifuren::cli(const int argc, const char* const argv[]) {
    if(argc <= 1) {
        // 没有参数表示使用GUI或者REST方式启动程序
        return false;
    }
    lifuren::message::registerMessageCallback(lifuren::message::Type::CLI_CONSOLE, messageCallback);
    for(int i = 0; i < argc; ++i) {
        SPDLOG_DEBUG("命令参数：{} {}", i, argv[i]);
    }
    const char* const command = argv[1];
    std::vector<std::string> args;
    for(int i = 2; i < argc; ++i) {
        args.push_back(argv[i]);
    }
    if(
        std::strcmp(command, "?")    == 0 ||
        std::strcmp(command, "help") == 0
    ) {
        ::help();
    } else if(std::strcmp(command, "audio") == 0) {
        ::generateAudio(args);
    } else if(std::strcmp(command, "video") == 0) {
        ::generateVideo(args);
    } else if(std::strcmp(command, "poetry") == 0) {
        ::generatePoetry(args);
    } else if(std::strcmp(command, "embedding") == 0) {
        ::embedding(args);
    } else {
        SPDLOG_WARN("不支持的命令：{}", command);
        ::help();
    }
    lifuren::message::unregisterMessageCallback(lifuren::message::Type::CLI_CONSOLE);
    return true;
}

// ./lifuren[.exe] audio [compose-shikuang] [pred|train] [model audio_file|dataset model_name]
static void generateAudio(const std::vector<std::string>& args) {
    if(args.size() < 4) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    const auto& client_name = args[0];
    auto client = lifuren::audio::getAudioClient(client_name);
    if(!client) {
        SPDLOG_WARN("没有终端类型：{}", client_name);
        return;
    }
    const std::string& type = args[1];
    if(type == "train") {
        const std::string& path       = args[2];
        const std::string& model_name = args[3];
        lifuren::config::ModelParams params {
            .model_name = model_name,
            .check_path = lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(),
            .train_path = lifuren::file::join({path, lifuren::config::DATASET_TRAIN}).string(),
            .val_path   = lifuren::file::join({path, lifuren::config::DATASET_VAL  }).string(),
            .test_path  = lifuren::file::join({path, lifuren::config::DATASET_TEST }).string(),
        };
        client->trainValAndTest(params);
        client->save(lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(), model_name + ".pt");
        SPDLOG_INFO("音频模型训练完成");
    } else if(type == "pred") {
        const std::string& model = args[2];
        const std::string& audio = args[3];
        const std::string output = audio + ".output.pcm";
        lifuren::audio::AudioParams params {
            .model  = model,
            .audio  = audio,
            .output = output
        };
        const auto [success, output_file] = client->pred(params);
        if(success) {
            SPDLOG_INFO("音频生成完成：{}", output_file);
        } else {
            SPDLOG_WARN("音频生成失败：{}", output_file);
        }
    } else {
        SPDLOG_WARN("无效类型：{}", type);
    }
}

// ./lifuren[.exe] video [paint-wudaozi] [pred|train] [model image_file|dataset model_name]
static void generateVideo(const std::vector<std::string>& args) {
    if(args.size() < 4) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    const auto& client_name = args[0];
    auto client = lifuren::video::getVideoClient(client_name);
    if(!client) {
        SPDLOG_WARN("没有终端类型：{}", client_name);
        return;
    }
    const std::string& type = args[1];
    if(type == "train") {
        const std::string& path       = args[2];
        const std::string& model_name = args[3];
        lifuren::config::ModelParams params {
            .model_name = model_name,
            .check_path = lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(),
            .train_path = lifuren::file::join({path, lifuren::config::DATASET_TRAIN}).string(),
            .val_path   = lifuren::file::join({path, lifuren::config::DATASET_VAL  }).string(),
            .test_path  = lifuren::file::join({path, lifuren::config::DATASET_TEST }).string(),
        };
        client->trainValAndTest(params);
        client->save(lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(), model_name + ".pt");
        SPDLOG_INFO("视频模型训练完成");
    } else if(type == "pred") {
        const std::string& model = args[2];
        const std::string& video = args[3];
        const std::string output = video + ".output.mp4";
        lifuren::video::VideoParams params {
            .model  = model,
            .video  = video,
            .output = output
        };
        const auto [success, output_file] = client->pred(params);
        if(success) {
            SPDLOG_INFO("视频生成完成：{}", output_file);
        } else {
            SPDLOG_WARN("视频生成失败：{}", output_file);
        }
    } else {
        SPDLOG_WARN("无效类型：{}", type);
    }
}

// ./lifuren[.exe] poetry [poetize-lidu|poetize-suxin] [pred|train] [model rhythm prompt1 prompt2|dataset model_name]
static void generatePoetry(const std::vector<std::string>& args) {
    if(args.size() < 4) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    const auto& client_name = args[0];
    auto client = lifuren::poetry::getPoetryClient(client_name);
    if(!client) {
        SPDLOG_WARN("没有终端类型：{}", client_name);
        return;
    }
    const std::string& type = args[1];
    if(type == "train") {
        const std::string& path       = args[2];
        const std::string& model_name = args[3];
        lifuren::config::ModelParams params {
            .model_name = model_name,
            .check_path = lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(),
            .train_path = lifuren::file::join({path, lifuren::config::DATASET_TRAIN}).string(),
            .val_path   = lifuren::file::join({path, lifuren::config::DATASET_VAL  }).string(),
            .test_path  = lifuren::file::join({path, lifuren::config::DATASET_TEST }).string(),
        };
        client->trainValAndTest(params);
        client->save(lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(), model_name + ".pt");
        SPDLOG_INFO("诗词模型训练完成");
    } else if(type == "pred") {
        if(args.size() < 5) {
            SPDLOG_WARN("缺少参数");
            return;
        }
        const std::string& model  = args[2];
        const std::string& rhythm = args[3];
        std::vector<std::string> prompts(args.begin() + 4, args.end());
        lifuren::poetry::PoetryParams params {
            .model   = model,
            .rhythm  = rhythm,
            .prompts = std::move(prompts)
        };
        const auto [success, result] = client->pred(params);
        if(success) {
            SPDLOG_INFO("诗词生成完成：{}", result);
        } else {
            SPDLOG_WARN("诗词生成失败：{}", result);
        }
    } else {
        SPDLOG_WARN("无效类型：{}", type);
    }
}

// ./lifuren[.exe] embedding dataset [audio|pepper|poetry] [faiss|elasticsearch] [pepper|ollama]
static void embedding(const std::vector<std::string>& args) {
    if(args.size() < 2) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    const auto& type = args[1];
    if(type == "audio") {
        embeddingAudio(args);
    } else if(type == "video") {
        embeddingPepper(args);
    } else if(type == "poetry") {
        embeddingPoetry(args);
    } else {
        SPDLOG_WARN("不支持的类型：{}", type);
    }
}

// ./lifuren[.exe] embedding dataset [audio|pepper|poetry]
static void embeddingAudio(const std::vector<std::string>& args) {
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    if(lifuren::audio::datasetPreprocessing(args[0])) {
        SPDLOG_INFO("音频嵌入成功");
    } else {
        SPDLOG_INFO("音频嵌入失败");
    }
}

// ./lifuren[.exe] embedding dataset [audio|pepper|poetry]
static void embeddingPepper(const std::vector<std::string>& args) {
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    if(lifuren::poetry::datasetPepperPreprocessing(args[0])) {
        SPDLOG_INFO("辣椒嵌入成功");
    } else {
        SPDLOG_WARN("辣椒嵌入失败");
    }
}

// ./lifuren[.exe] embedding dataset [audio|pepper|poetry] [faiss|elasticsearch] [pepper|ollama]
static void embeddingPoetry(const std::vector<std::string>& args) {
    if(args.size() < 4) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    const auto& dataset        = args[0];
    const auto& rag_type       = args[2];
    const auto& embedding_type = args[3];
    if(lifuren::poetry::datasetPoetryPreprocessing(dataset, rag_type, embedding_type)) {
        SPDLOG_INFO("诗词嵌入成功");
    } else {
        SPDLOG_WARN("诗词嵌入失败");
    }
}

static void help() {
    std::cout << R"(
./lifuren[.exe] 命令 [参数...]
./lifuren[.exe] video     [paint-wudaozi             ] [pred|train] [model image_file|dataset model_name]
./lifuren[.exe] audio     [compose-shikuang          ] [pred|train] [model audio_file|dataset model_name]
./lifuren[.exe] poetry    [poetize-lidu|poetize-suxin] [pred|train] [model rhythm prompt1 prompt2|dataset model_name]
./lifuren[.exe] embedding dataset [audio|pepper|poetry] [faiss|elasticsearch] [pepper|ollama]
./lifuren[.exe] [?|help]
)" << std::endl;
}

static void messageCallback(bool finish, const char* message) {
    #if defined(_DEBUG) || !defined(NDEBUG)
    // 测试时控制台日志已经打开
    #else
    std::cout << message << std::endl;
    #endif
}
