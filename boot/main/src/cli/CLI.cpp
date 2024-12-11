#include "lifuren/CLI.hpp"

#include <mutex>
#include <string>
#include <vector>
#include <iostream>
#include <condition_variable>

#include "spdlog/spdlog.h"

#include "lifuren/RAG.hpp"
#include "lifuren/File.hpp"
#include "lifuren/Torch.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Dataset.hpp"
#include "lifuren/Message.hpp"
#include "lifuren/audio/Audio.hpp"
#include "lifuren/EmbeddingClient.hpp"
#include "lifuren/video/ActClient.hpp"
#include "lifuren/image/PaintClient.hpp"
#include "lifuren/audio/ComposeClient.hpp"
#include "lifuren/poetry/PoetizeClient.hpp"

static void act      (const std::vector<std::string>&); // 视频生成
static void paint    (const std::vector<std::string>&); // 图片生成
static void compose  (const std::vector<std::string>&); // 音频生成
static void poetize  (const std::vector<std::string>&); // 诗词生成
static void pcm      (const std::vector<std::string>&); // 转为PCM
static void pepper   (const std::vector<std::string>&); // 辣椒嵌入
static void embedding(const std::vector<std::string>&); // 诗词嵌入
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
        help();
    } else if(std::strcmp(command, "act") == 0) {
        act(args);
    } else if(std::strcmp(command, "paint") == 0) {
        paint(args);
    } else if(std::strcmp(command, "compose") == 0) {
        compose(args);
    } else if(std::strcmp(command, "poetize") == 0) {
        poetize(args);
    } else if(std::strcmp(command, "pcm") == 0) {
        pcm(args);
    } else if(std::strcmp(command, "pepper") == 0) {
        pepper(args);
    } else if(std::strcmp(command, "embedding") == 0) {
        embedding(args);
    } else {
        SPDLOG_WARN("不支持的命令：{}", command);
    }
    return true;
}

static void act(const std::vector<std::string>& args) {
    if(args.size() < 4) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = lifuren::getActClient(args[0]);
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
        lifuren::ActParams params {
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

static void paint(const std::vector<std::string>& args) {
    if(args.size() < 4) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = lifuren::getPaintClient(args[0]);
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
        const std::string& image = args[3];
        const std::string output = image + ".output.jpg";
        lifuren::PaintParams params {
            .model  = model,
            .image  = image,
            .output = output
        };
        client->pred(params);
    } else {
        SPDLOG_WARN("无效类型：{}", type);
        return;
    }
}

static void compose(const std::vector<std::string>& args) {
    if(args.size() < 4) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = lifuren::getComposeClient(args[0]);
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
        lifuren::ComposeParams params {
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

static void poetize(const std::vector<std::string>& args) {
    if(args.size() < 4) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = lifuren::getPoetizeClient(args[0]);
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
        lifuren::PoetizeParams params {
            .model   = model,
            .rhythm  = rhythm,
            .prompts = prompts
        };
        std::string result = client->pred(params);
        lifuren::message::sendMessage(result.c_str());
    } else {
        SPDLOG_WARN("无效类型：{}", type);
        return;
    }
}

static void pcm(const std::vector<std::string>& args) {
    lifuren::message::registerMessageCallback(lifuren::message::Type::AUDIO_AUDIO_TO_PCM, messageCallback);
    if(args.empty()) {
        lifuren::message::sendMessage("缺少参数");
        return;
    }
    auto preprocessing = std::bind(&lifuren::audio::preprocessing, std::placeholders::_1);
    if(lifuren::dataset::allDatasetPreprocessing(args[0], preprocessing)) {
        lifuren::message::sendMessage("PCM转换成功");
    } else {
        lifuren::message::sendMessage("PCM转换失败");
    }
    lifuren::message::unregisterMessageCallback(lifuren::message::Type::AUDIO_AUDIO_TO_PCM);
}

static void pepper(const std::vector<std::string>& args) {
    if(args.empty()) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    auto client = std::make_unique<lifuren::PepperEmbeddingClient>();
    auto embedding = std::bind(&lifuren::PepperEmbeddingClient::embedding, client.get(), std::placeholders::_1);
    if(lifuren::dataset::allDatasetPreprocessing(args[0], embedding)) {
        SPDLOG_INFO("辣椒嵌入成功");
    } else {
        SPDLOG_WARN("辣椒嵌入失败");
    }
}

static void embedding(const std::vector<std::string>& args) {
    if(args.size() < 3) {
        SPDLOG_WARN("缺少参数");
        return;
    }
    if(lifuren::RAGClient::rag(args[0], args[1], args[2])) {
        SPDLOG_INFO("嵌入成功");
    } else {
        SPDLOG_WARN("嵌入失败");
    }
}

static void help() {
    std::cout << R"(
./lifuren[.exe] 命令 [参数...]
./lifuren[.exe] act       [act-tangxianzu  |act-guanhanqing  ] [train|pred] [model video_file|dataset model_name]
./lifuren[.exe] paint     [paint-wudaozi   |paint-gukaizhi   ] [train|pred] [model image_file|dataset model_name]
./lifuren[.exe] compose   [compose-shikuang|compose-liguinian] [train|pred] [model audio_file|dataset model_name]
./lifuren[.exe] poetize   [poetize-lidu    |poetize-suxin    ] [train|pred] [model rhythm prompt1 prompt2|dataset model_name]
./lifuren[.exe] pcm       dataset
./lifuren[.exe] pepper    dataset
./lifuren[.exe] embedding [faiss|elasticsearch] dataset [pepper|ollama]
./lifuren[.exe] [?|help]
    )" << std::endl;
}

static void messageCallback(bool finish, const char* message) {
    #if defined(_DEBUG) || !defined(NDEBUG)
    #else
    std::cout << message << std::endl;
    #endif
}
