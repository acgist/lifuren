#include "lifuren/CLI.hpp"

#include <mutex>
#include <string>
#include <vector>
#include <iostream>
#include <condition_variable>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/Message.hpp"
#include "lifuren/RAGClient.hpp"
#include "lifuren/audio/Audio.hpp"
#include "lifuren/video/Video.hpp"
#include "lifuren/poetry/Poetry.hpp"
#include "lifuren/EmbeddingClient.hpp"
#include "lifuren/audio/AudioClient.hpp"
#include "lifuren/video/VideoClient.hpp"
#include "lifuren/poetry/PoetryClient.hpp"

static void audio (const std::vector<std::string>&); // 音频生成
static void video (const std::vector<std::string>&); // 视频生成
static void poetry(const std::vector<std::string>&); // 诗词生成
static void embedding      (const std::vector<std::string>&); // 数据嵌入
static void embeddingAudio (const std::vector<std::string>&); // 音频嵌入
static void embeddingPepper(const std::vector<std::string>&); // 辣椒嵌入
static void embeddingPoetry(const std::vector<std::string>&); // 诗词嵌入
static void help(); // 帮助
static void messageCallback(bool, const char*); // 消息回调

// TODO: 注册回调

bool lifuren::cli(const int argc, const char* const argv[]) {
    if(argc <= 1) {
        return false;
    }
    for(int i = 0; i < argc; ++i) {
        SPDLOG_DEBUG("命令参数：{}", argv[i]);
    }
    // 命令
    const char* const command = argv[1];
    // 参数
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
        ::audio(args);
    } else if(std::strcmp(command, "video") == 0) {
        ::video(args);
    } else if(std::strcmp(command, "poetry") == 0) {
        ::poetry(args);
    } else if(std::strcmp(command, "embedding") == 0) {
        ::embedding(args);
    } else {
        SPDLOG_WARN("不支持的命令：{}", command);
    }
    return true;
}

static void audio(const std::vector<std::string>& args) {
    if(args.size() < 4) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = lifuren::getAudioClient(args[0]);
    if(!client) {
        SPDLOG_WARN("没有终端类型：{}", args[0]);
        return;
    }
    const std::string& type = args[1];
    if(type == "train") {
        const std::string& path = args[2];
        const std::string& model_name = args[3];
        lifuren::config::ModelParams params {
            .check_path = lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(),
            .model_name = model_name,
            .train_path = lifuren::file::join({path, lifuren::config::DATASET_TRAIN}).string(),
            .val_path   = lifuren::file::join({path, lifuren::config::DATASET_VAL}).string(),
            .test_path  = lifuren::file::join({path, lifuren::config::DATASET_TEST}).string(),
        };
        client->trainValAndTest(params);
        client->save(lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(), model_name + ".pt");
    } else if(type == "pred") {
        const std::string& model = args[2];
        const std::string& audio = args[3];
        const std::string output = audio + ".output.pcm";
        lifuren::AudioParams params {
            .model  = model,
            .audio  = audio,
            .output = output
        };
        client->pred(params);
        lifuren::audio::toFile(output);
    } else {
        SPDLOG_WARN("无效类型：{}", type);
        return;
    }
}

static void video(const std::vector<std::string>& args) {
    if(args.size() < 4) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = lifuren::getVideoClient(args[0]);
    if(!client) {
        SPDLOG_WARN("没有终端类型：{}", args[0]);
        return;
    }
    const std::string& type = args[1];
    if(type == "train") {
        const std::string& path = args[2];
        const std::string& model_name = args[3];
        lifuren::config::ModelParams params {
            .check_path = lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(),
            .model_name = model_name,
            .train_path = lifuren::file::join({path, lifuren::config::DATASET_TRAIN}).string(),
            .val_path   = lifuren::file::join({path, lifuren::config::DATASET_VAL}).string(),
            .test_path  = lifuren::file::join({path, lifuren::config::DATASET_TEST}).string(),
        };
        client->trainValAndTest(params);
        client->save(lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(), model_name + ".pt");
    } else if(type == "pred") {
        const std::string& model = args[2];
        const std::string& video = args[3];
        const std::string output = video + ".output.mp4";
        lifuren::VideoParams params {
            .model  = model,
            .video  = video,
            .output = output
        };
        client->pred(params);
    } else {
        SPDLOG_WARN("无效类型：{}", type);
        return;
    }
}

static void poetry(const std::vector<std::string>& args) {
    if(args.size() < 4) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = lifuren::getPoetryClient(args[0]);
    if(!client) {
        SPDLOG_WARN("没有终端类型：{}", args[0]);
        return;
    }
    const std::string& type = args[1];
    if(type == "train") {
        const std::string& path = args[2];
        const std::string& model_name = args[3];
        lifuren::config::ModelParams params {
            .check_path = lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(),
            .model_name = model_name,
            .train_path = lifuren::file::join({path, lifuren::config::DATASET_TRAIN}).string(),
            .val_path   = lifuren::file::join({path, lifuren::config::DATASET_VAL}).string(),
            .test_path  = lifuren::file::join({path, lifuren::config::DATASET_TEST}).string(),
        };
        client->trainValAndTest(params);
        client->save(lifuren::file::join({path, lifuren::config::LIFUREN_HIDDEN_FILE}).string(), model_name + ".pt");
    } else if(type == "pred") {
        if(args.size() < 5) {
            SPDLOG_WARN("缺少参数");
            return;
        }
        const std::string& model = args[2];
        const std::string& rhythm = args[3];
        std::vector<std::string> prompts(args.begin() + 4, args.end());
        lifuren::PoetryParams params {
            .model   = model,
            .rhythm  = rhythm,
            .prompts = prompts
        };
        std::string result = client->pred(params);
        SPDLOG_INFO("{}", result);
    } else {
        SPDLOG_WARN("无效类型：{}", type);
        return;
    }
}

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
        return;
    }
}

static void embeddingAudio(const std::vector<std::string>& args) {
    lifuren::message::registerMessageCallback(lifuren::message::Type::AUDIO_EMBEDDING, messageCallback);
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    if(lifuren::dataset::allDatasetPreprocessing(args[0], lifuren::config::EMBEDDING_MODEL_FILE, &lifuren::audio::embedding)) {
        SPDLOG_INFO("音频嵌入成功");
    } else {
        SPDLOG_INFO("音频嵌入失败");
    }
    lifuren::message::unregisterMessageCallback(lifuren::message::Type::AUDIO_EMBEDDING);
}

static void embeddingPepper(const std::vector<std::string>& args) {
    lifuren::message::registerMessageCallback(lifuren::message::Type::POETRY_EMBEDDING_PEPPER, messageCallback);
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    if(lifuren::dataset::allDatasetPreprocessing(args[0], lifuren::config::PEPPER_MODEL_FILE, &lifuren::poetry::pepper::embedding, true)) {
        SPDLOG_INFO("辣椒嵌入成功");
    } else {
        SPDLOG_WARN("辣椒嵌入失败");
    }
    lifuren::message::unregisterMessageCallback(lifuren::message::Type::POETRY_EMBEDDING_PEPPER);
}

static void embeddingPoetry(const std::vector<std::string>& args) {
    if(args.size() < 4) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    std::shared_ptr<lifuren::RAGClient> client = std::move(lifuren::RAGClient::getClient(args[2], args[0], args[3]));
    // std::function<bool(const std::string&, const std::string&, std::ofstream&, lifuren::thread::ThreadPool&)>
    auto embedding = std::bind(&lifuren::rag::embedding, client, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
    if(lifuren::dataset::allDatasetPreprocessing(args[0], lifuren::config::EMBEDDING_MODEL_FILE, embedding)) {
        SPDLOG_INFO("诗词嵌入成功");
    } else {
        SPDLOG_WARN("诗词嵌入失败");
    }
}

static void help() {
    std::cout << R"(
./lifuren[.exe] 命令 [参数...]
./lifuren[.exe] paint     [paint-wudaozi                ] [pred|train] [model image_file|dataset model_name]
./lifuren[.exe] compose   [compose-shikuang             ] [pred|train] [model audio_file|dataset model_name]
./lifuren[.exe] poetize   [poetize-lidu | poetize-suxin ] [pred|train] [model rhythm prompt1 prompt2|dataset model_name]
./lifuren[.exe] embedding dataset [audio|pepper|poetry] [faiss|elasticsearch] [pepper|ollama]
./lifuren[.exe] [?|help]
    )" << std::endl;
}

static void messageCallback(bool finish, const char* message) {
    #if defined(_DEBUG) || !defined(NDEBUG)
    #else
    std::cout << message << std::endl;
    #endif
}
