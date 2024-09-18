/**
 * https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/cli/main.cpp
 */
#include "lifuren/PaintClient.hpp"

#include <thread>
#include <random>

#include "spdlog/spdlog.h"

#include "stable-diffusion.h"
#include "thirdparty/stb_image_resize.h"

#include "opencv2/opencv.hpp"

#include "lifuren/Raii.hpp"
#include "lifuren/Files.hpp"
#include "lifuren/Images.hpp"
#include "lifuren/Lifuren.hpp"

const char* options_rng[] {
    "std_default",
    "cuda",
};

const char* options_mode[] {
    "txt2img",
    "img2img",
    "img2vid",
    "convert",
};

const char* options_wtype[] {
    "f32",
    "f16",
    "q4_0",
    "q4_1",
    "q4_2",
    "q4_3",
    "q5_0",
    "q5_1",
    "q8_0",
    "q8_1",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "q8_k",
    "iq2_xxs",
    "iq2_xs",
    "iq3_xxs",
    "iq1_s",
    "iq4_nl",
    "iq3_s",
    "iq2_s",
    "iq4_xs",
    "i8",
    "i16",
    "i32",
    "i64",
    "f64",
    "iq1_m",
    "bf16",
    "q4_0_4_4",
    "q4_0_4_8",
    "q4_0_8_8"
};

const char* options_schedule[] {
    "default",
    "discrete",
    "karras",
    "ays",
};

const char* options_sample_method[] {
    "euler_a",
    "euler",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "lcm",
};

enum class SDMode {
    TXT2IMG,
    IMG2IMG,
    IMG2VID,
    CONVERT,
};

struct SDParams {
    SDMode          mode          = SDMode::TXT2IMG; // -M, --mode [MODEL]
    sd_type_t       wtype         = SD_TYPE_COUNT;   // --type [TYPE]
    rng_type_t      rng_type      = STD_DEFAULT_RNG; // --rng
    schedule_t      schedule      = DEFAULT;         // --schedule
    sample_method_t sample_method = EULER_A;         // --sampling-method

    std::string prompt;                     // -p, --prompt [PROMPT]
    std::string vae_path;                   // --vae [VAE]
    std::string model_path;                 // -m, --model [MODEL]
    std::string input_path;                 // -i, --init-img [IMAGE]
    std::string t5xxl_path;                 // --t5xxl
    std::string taesd_path;                 // --taesd [TAESD_PATH]
    std::string clip_l_path;                // --clip_l
    std::string esrgan_path;                // --upscale-model [ESRGAN_PATH]
    std::string output_path = "output.png"; // -o, --output OUTPUT
    std::string lora_model_dir;             // --lora-model-dir [DIR]
    std::string controlnet_path;            // --control-net [CONTROL_PATH]
    std::string embeddings_path;            // --embd-dir [EMBEDDING_PATH]
    std::string negative_prompt;            // -n, --negative-prompt PROMPT
    std::string control_image_path;         // --control-image [IMAGE]
    std::string input_id_images_path;       // --input-id-images-dir [DIR]
    std::string diffusion_model_path;       // --diffusion-model
    std::string stacked_id_embeddings_path; // --stacked-id-embd-dir [DIR]

    int fps              = 6;   // -
    int width            = 512; // -W, --width W
    int height           = 512; // -H, --height H
    int n_threads        = -1;  // -t, --threads N
    int clip_skip        = -1;  // --clip-skip N
    int batch_count      = 1;   // -b, --batch-count COUNT
    int video_frames     = 6;   // -
    int sample_steps     = 20;  // --steps STEPS
    int upscale_repeats  = 1;   // --upscale-repeats
    int motion_bucket_id = 127; // 

    int64_t seed = 42; // -s SEED, --seed SEED

    float min_cfg            = 1.0F;  // -
    float strength           = 0.75F; // --strength STRENGTH
    float guidance           = 3.5F;  // --guidance
    float cfg_scale          = 7.0F;  // --cfg-scale SCALE
    float style_ratio        = 20.0F; // --style-ratio STYLE-RATIO
    float control_strength   = 0.9F;  // --control-strength STRENGTH
    float augmentation_level = 0.0F;  // -

    bool color            = false; // --color
    bool verbose          = false; // -v, --verbose
    bool vae_tiling       = false; // --vae-tiling
    bool vae_on_cpu       = false; // --vae-on-cpu
    bool clip_on_cpu      = false; // --clip-on-cpu
    bool normalize_input  = false; // --normalize-input
    bool control_net_cpu  = false; // --control-net-cpu
    bool canny_preprocess = false; // --canny
};

#if LFR_SHARE_SD_CTX
static bool shareSDCtx = true;
#else
static bool shareSDCtx = false;
#endif

sd_ctx_t* share_sd_ctx{ nullptr };

static void logCallback(sd_log_level_t level, const char* log, void* data);
static void initSDParams(SDParams&  params, const lifuren::PaintClient::PaintOptions& options);
static void printSDParams(SDParams& params);
static bool checkSDParams(SDParams& params);
static void setOptions(int&         target, const std::map<std::string, std::string>& options, const std::string& key);
static void setOptions(bool&        target, const std::map<std::string, std::string>& options, const std::string& key);
static void setOptions(float&       target, const std::map<std::string, std::string>& options, const std::string& key);
static void setOptions(int64_t&     target, const std::map<std::string, std::string>& options, const std::string& key);
static void setOptions(std::string& target, const std::map<std::string, std::string>& options, const std::string& key);
static int  getOptions(int count, const char** mapping, const std::map<std::string, std::string>& options, const std::string& key, const int defaultValue = 0);
static sd_image_t* loadInputImage  (SDParams& params);
static sd_image_t* loadControlImage(SDParams& params);
static bool paintTxt2Img(SDParams& params, sd_ctx_t* sd_ctx);
static bool paintImg2Img(SDParams& params, sd_ctx_t* sd_ctx);
static bool paintImg2Vid(SDParams& params, sd_ctx_t* sd_ctx);
static bool writeImg(SDParams& params, size_t count, sd_image_t* result);
static void releaseImg(sd_image_t** image);
static sd_ctx_t* getSDCtx(SDParams& params, bool vae_decode_only);

lifuren::StableDiffusionCPPPaintClient::StableDiffusionCPPPaintClient() {
}

lifuren::StableDiffusionCPPPaintClient::~StableDiffusionCPPPaintClient() {
}

bool lifuren::StableDiffusionCPPPaintClient::paint(const PaintOptions& options, lifuren::PaintClient::PaintCallback callback) {
    std::unique_lock<std::mutex> lock(this->mutex);
    this->running = true;
    lifuren::Finally finally([this]() {
        this->running = false;
    });
    SDParams params{};
    initSDParams(params, options);
    printSDParams(params);
    sd_set_log_callback(logCallback, static_cast<void*>(&params));
    if(!checkSDParams(params)) {
        return false;
    }
    if(params.mode == SDMode::CONVERT) {
        SPDLOG_INFO("模型转换：{} - {}", params.model_path, params.output_path);
        return convert(params.model_path.c_str(), params.vae_path.c_str(), params.output_path.c_str(), params.wtype);
    } else {
        bool vae_decode_only = true;
        if (params.mode == SDMode::IMG2IMG || params.mode == SDMode::IMG2VID) {
            vae_decode_only = false;
        }
        sd_ctx_t* sd_ctx = getSDCtx(params, vae_decode_only);
        if (sd_ctx == NULL) {
            SPDLOG_WARN("加载模型失败");
            return false;
        }
        bool success = false;
        if(params.mode == SDMode::TXT2IMG) {
            success = paintTxt2Img(params, sd_ctx);
        } else if(params.mode == SDMode::IMG2IMG) {
            success = paintImg2Img(params, sd_ctx);
        } else if(params.mode == SDMode::IMG2VID) {
            success = paintImg2Vid(params, sd_ctx);
        } else {
            SPDLOG_WARN("不支持的类型");
        }
        if(shareSDCtx) {
            // 共享
        } else {
            free_sd_ctx(sd_ctx);
        }
        return success;
    }
}

bool lifuren::StableDiffusionCPPPaintClient::stop() {
    // TODO: 停止
    return true;
}

static void logCallback(sd_log_level_t level, const char* log, void* data) {
    std::string message = log;
    message.resize(message.size() - 1);
    SPDLOG_DEBUG("SD Log : {}", message);
}

static void initSDParams(SDParams& params, const lifuren::PaintClient::PaintOptions& paintOptions) {
    const auto& config      = lifuren::config::CONFIG.stableDiffusionCPP;
    const auto& options     = config.options;
    const auto& imageConfig = lifuren::config::CONFIG.image;

    params.mode          = static_cast<SDMode>(static_cast<int>(paintOptions.mode));
    params.wtype         = static_cast<sd_type_t>(      getOptions(sd_type_t::SD_TYPE_COUNT,          options_wtype,         options, "wtype",         sd_type_t::SD_TYPE_F32));
    params.rng_type      = static_cast<rng_type_t>(     getOptions(rng_type_t::CUDA_RNG + 1,          options_rng,           options, "rng_type",      rng_type_t::STD_DEFAULT_RNG));
    params.schedule      = static_cast<schedule_t>(     getOptions(schedule_t::N_SCHEDULES,           options_schedule,      options, "schedule",      schedule_t::DEFAULT));
    params.sample_method = static_cast<sample_method_t>(getOptions(sample_method_t::N_SAMPLE_METHODS, options_sample_method, options, "sample_method", sample_method_t::EULER_A));
    
    setOptions(params.prompt,                     options, "prompt");
    setOptions(params.vae_path,                   options, "vae_path");
    setOptions(params.model_path,                 options, "model_path");
    setOptions(params.input_path,                 options, "input_path");
    setOptions(params.t5xxl_path,                 options, "t5xxl_path");
    setOptions(params.taesd_path,                 options, "taesd_path");
    setOptions(params.clip_l_path,                options, "clip_l_path");
    setOptions(params.esrgan_path,                options, "esrgan_path");
    setOptions(params.output_path,                options, "output_path");
    setOptions(params.lora_model_dir,             options, "lora_model_dir");
    setOptions(params.controlnet_path,            options, "controlnet_path");
    setOptions(params.embeddings_path,            options, "embeddings_path");
    setOptions(params.negative_prompt,            options, "negative_prompt");
    setOptions(params.control_image_path,         options, "control_image_path");
    setOptions(params.input_id_images_path,       options, "input_id_images_path");
    setOptions(params.diffusion_model_path,       options, "diffusion_model_path");
    setOptions(params.stacked_id_embeddings_path, options, "stacked_id_embeddings_path");

    setOptions(params.fps,              options, "fps");
    setOptions(params.width,            options, "width");
    setOptions(params.height,           options, "height");
    setOptions(params.n_threads,        options, "n_threads");
    setOptions(params.clip_skip,        options, "clip_skip");
    setOptions(params.batch_count,      options, "batch_count");
    setOptions(params.video_frames,     options, "video_frames");
    setOptions(params.sample_steps,     options, "sample_steps");
    setOptions(params.upscale_repeats,  options, "upscale_repeats");
    setOptions(params.motion_bucket_id, options, "motion_bucket_id");

    setOptions(params.seed, options, "seed");

    setOptions(params.min_cfg,            options, "min_cfg");
    setOptions(params.strength,           options, "strength");
    setOptions(params.guidance,           options, "guidance");
    setOptions(params.cfg_scale,          options, "cfg_scale");
    setOptions(params.style_ratio,        options, "style_ratio");
    setOptions(params.control_strength,   options, "control_strength");
    setOptions(params.augmentation_level, options, "augmentation_level");

    setOptions(params.color,            options, "color");
    setOptions(params.verbose,          options, "verbose");
    setOptions(params.vae_tiling,       options, "vae_tiling");
    setOptions(params.vae_on_cpu,       options, "vae_on_cpu");
    setOptions(params.clip_on_cpu,      options, "clip_on_cpu");
    setOptions(params.normalize_input,  options, "normalize_input");
    setOptions(params.control_net_cpu,  options, "control_net_cpu");
    setOptions(params.canny_preprocess, options, "canny_preprocess");

    if(!config.model.empty()) {
        params.model_path = config.model;
    }
    if(!imageConfig.output.empty()) {
        params.output_path = imageConfig.output;
    }

    if(!paintOptions.image.empty()) {
        params.input_path = paintOptions.image;
    }
    if(!paintOptions.model.empty()) {
        params.model_path = paintOptions.model;
    }
    if(!paintOptions.prompt.empty()) {
        params.prompt = paintOptions.prompt;
    }
    if(!paintOptions.output.empty()) {
        params.output_path = paintOptions.output;
    }
    if(paintOptions.seed > 0) {
        params.seed = paintOptions.seed;
    }
    if(paintOptions.count > 0) {
        params.batch_count = paintOptions.count;
    }
    if(paintOptions.steps > 0) {
        params.sample_steps = paintOptions.steps;
    }
    if(paintOptions.width > 0) {
        params.width = paintOptions.width;
    }
    if(paintOptions.height > 0) {
        params.height = paintOptions.height;
    }
    params.color = paintOptions.color;

    if (params.seed <= 0) {
        std::random_device device{};
        std::mt19937 random{device()};
        params.seed = random();
    }
    if (params.n_threads <= 0) {
        params.n_threads = get_num_physical_cores();
        // params.n_threads = std::thread::hardware_concurrency()
    }
    if (params.mode == SDMode::CONVERT) {
        if (params.output_path == "output.png") {
            params.output_path = "output.gguf";
        }
    }
}

static void printSDParams(SDParams& params) {
    SPDLOG_DEBUG(R"(
SD Params:
    mode                      :    {}
    wtype                     :    {}
    rng_type                  :    {}
    schedule                  :    {}
    sample_method             :    {}

    prompt                    :    {}
    vae_path                  :    {}
    model_path                :    {}
    input_path                :    {}
    t5xxl_path                :    {}
    taesd_path                :    {}
    clip_l_path               :    {}
    esrgan_path               :    {}
    output_path               :    {}
    lora_model_dir            :    {}
    controlnet_path           :    {}
    embeddings_path           :    {}
    negative_prompt           :    {}
    control_image_path        :    {}
    input_id_images_path      :    {}
    diffusion_model_path      :    {}
    stacked_id_embeddings_path:    {}

    fps                       :    {}
    width                     :    {}
    height                    :    {}
    n_threads                 :    {}
    clip_skip                 :    {}
    batch_count               :    {}
    video_frames              :    {}
    sample_steps              :    {}
    upscale_repeats           :    {}
    motion_bucket_id          :    {}

    seed                      :    {}

    min_cfg                   :    {:.2f}
    strength                  :    {:.2f}
    guidance                  :    {:.2f}
    cfg_scale                 :    {:.2f}
    style ratio               :    {:.2f}
    control_strength          :    {:.2f}
    augmentation_level        :    {:.2f}
    
    color                     :    {}
    verbose                   :    {}
    vae_tiling                :    {}
    vae_on_cpu                :    {}
    clip_on_cpu               :    {}
    normalize_input           :    {}
    control_net_cpu           :    {}
    canny_preprocess          :    {}
    )",
    options_mode[static_cast<int>(params.mode)],
    options_wtype[params.wtype],
    options_rng[params.rng_type],
    options_schedule[params.schedule],
    options_sample_method[params.sample_method],

    params.prompt,
    params.vae_path,
    params.model_path,
    params.input_path,
    params.t5xxl_path,
    params.taesd_path,
    params.clip_l_path,
    params.esrgan_path,
    params.output_path,
    params.lora_model_dir,
    params.controlnet_path,
    params.embeddings_path,
    params.negative_prompt,
    params.control_image_path,
    params.input_id_images_path,
    params.diffusion_model_path,
    params.stacked_id_embeddings_path,

    params.fps,
    params.width,
    params.height,
    params.n_threads,
    params.clip_skip,
    params.batch_count,
    params.video_frames,
    params.sample_steps,
    params.upscale_repeats,
    params.motion_bucket_id,

    params.seed,

    params.min_cfg,
    params.strength,
    params.guidance,
    params.cfg_scale,
    params.style_ratio,
    params.control_strength,
    params.augmentation_level,

    params.color,
    params.verbose,
    params.vae_tiling,
    params.vae_on_cpu,
    params.clip_on_cpu,
    params.normalize_input,
    params.control_net_cpu,
    params.canny_preprocess
    );
}

static bool checkSDParams(SDParams& params) {
    if (params.mode == SDMode::TXT2IMG && params.prompt.empty()) {
        SPDLOG_WARN("提示内容为空（prompt）");
        return false;
    }
    if (params.mode == SDMode::IMG2IMG && params.input_path.empty()) {
        SPDLOG_WARN("文件内容为空（input_path）");
        return false;
    }
    if (params.mode == SDMode::IMG2VID && params.input_path.empty()) {
        SPDLOG_WARN("文件内容为空（input_path）");
        return false;
    }
    if (params.width <= 0 || params.width % 64 != 0) {
        SPDLOG_WARN("参数错误（width） = {}", params.width);
        return false;
    }
    if (params.height <= 0 || params.height % 64 != 0) {
        SPDLOG_WARN("参数错误（height） = {}", params.height);
        return false;
    }
    if (params.strength < 0.0F || params.strength > 1.0F) {
        SPDLOG_WARN("参数错误（strength） = {}", params.strength);
        return false;
    }
    if (params.model_path.empty() && params.diffusion_model_path.empty()) {
        SPDLOG_WARN("模型路径为空（diffusion_model_path）");
        return false;
    }
    if (params.output_path.empty()) {
        SPDLOG_WARN("输出目录为空（output_path）");
        return false;
    }
    if (params.sample_steps <= 0) {
        SPDLOG_WARN("参数错误（sample_steps） = {}", params.sample_steps);
        return false;
    }
    return true;
}

static void setOptions(int& target, const std::map<std::string, std::string>& options, const std::string& key) {
    auto value = options.find(key);
    if(value == options.end()) {
        return;
    }
    target = std::stoi(value->second);
}

static void setOptions(bool& target, const std::map<std::string, std::string>& options, const std::string& key) {
    auto value = options.find(key);
    if(value == options.end()) {
        return;
    }
    target = !strcmp(value->second.c_str(), "true");
}

static void setOptions(float& target, const std::map<std::string, std::string>& options, const std::string& key) {
    auto value = options.find(key);
    if(value == options.end()) {
        return;
    }
    target = std::stof(value->second);
}

static void setOptions(int64_t& target, const std::map<std::string, std::string>& options, const std::string& key) {
    auto value = options.find(key);
    if(value == options.end()) {
        return;
    }
    target = std::stoll(value->second);
}

static void setOptions(std::string& target, const std::map<std::string, std::string>& options, const std::string& key) {
    auto value = options.find(key);
    if(value == options.end()) {
        return;
    }
    target = value->second;
}

static int getOptions(int count, const char** mapping, const std::map<std::string, std::string>& options, const std::string& key, const int defaultValue) {
    std::string value;
    setOptions(value, options, key);
    if(value.empty()) {
        return defaultValue;
    }
    const char* selected = value.c_str();
    for (int index = 0; index < count; ++index) {
        if (!strcmp(selected, mapping[index])) {
            return index;
        }
    }
    return defaultValue;
}

static sd_image_t* loadInputImage(SDParams& params) {
    sd_image_t* input_image     { nullptr };
    uint8_t   * input_image_data{ nullptr };
    size_t width { 0 };
    size_t height{ 0 };
    size_t length{ 0 };
    lifuren::images::read(params.input_path, &input_image_data, width, height, length);
    if (static_cast<size_t>(params.width) != width || static_cast<size_t>(params.height) != height) {
        const int resized_width  = params.width;
        const int resized_height = params.height;
        uint8_t* resized_image_data = new uint8_t[resized_width * resized_height * 3];
        stbir_resize(
            input_image_data,   width,         height,         0,
            resized_image_data, resized_width, resized_height, 0,
            STBIR_TYPE_UINT8, 3, STBIR_ALPHA_CHANNEL_NONE, 0,
            STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
            STBIR_FILTER_BOX, STBIR_FILTER_BOX,
            STBIR_COLORSPACE_SRGB, nullptr
        );
        delete[] input_image_data;
        input_image_data = nullptr;
        input_image = new sd_image_t{static_cast<uint32_t>(params.width), static_cast<uint32_t>(params.height), 3, resized_image_data};
    } else {
        input_image = new sd_image_t{static_cast<uint32_t>(params.width), static_cast<uint32_t>(params.height), 3, input_image_data};
    }
    return input_image;
}

static sd_image_t* loadControlImage(SDParams& params) {
    sd_image_t* control_image     { nullptr };
    uint8_t   * control_image_data{ nullptr };
    if (!params.controlnet_path.empty() && !params.control_image_path.empty()) {
        size_t width {0};
        size_t height{0};
        size_t length{0};
        lifuren::images::read(params.control_image_path, &control_image_data, width, height, length);
        control_image = new sd_image_t{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 3, control_image_data};
        if (params.canny_preprocess) {
            uint8_t* canny_data = preprocess_canny(
                control_image->data,
                control_image->width,
                control_image->height,
                0.08F,
                0.08F,
                0.8F,
                1.0F,
                false
            );
            memcpy(control_image_data, canny_data, length);
            free(canny_data);
            canny_data = NULL;
        }
    }
    return control_image;
}

static bool paintTxt2Img(SDParams& params, sd_ctx_t* sd_ctx) {
    sd_image_t* control_image { loadControlImage(params) };
    sd_image_t* result = txt2img(
        sd_ctx,
        params.prompt.c_str(),
        params.negative_prompt.c_str(),
        params.clip_skip,
        params.cfg_scale,
        params.guidance,
        params.width,
        params.height,
        params.sample_method,
        params.sample_steps,
        params.seed,
        params.batch_count,
        control_image,
        params.control_strength,
        params.style_ratio,
        params.normalize_input,
        params.input_id_images_path.c_str()
    );
    bool ret = writeImg(params, params.batch_count, result);
    releaseImg(&control_image);
    free(result);
    result = NULL;
    return ret;
}

static bool paintImg2Img(SDParams& params, sd_ctx_t* sd_ctx) {
    sd_image_t* input_image   { loadInputImage(params) };
    sd_image_t* control_image { loadControlImage(params) };
    sd_image_t* result = img2img(
        sd_ctx,
        *input_image,
        params.prompt.c_str(),
        params.negative_prompt.c_str(),
        params.clip_skip,
        params.cfg_scale,
        params.guidance,
        params.width,
        params.height,
        params.sample_method,
        params.sample_steps,
        params.strength,
        params.seed,
        params.batch_count,
        control_image,
        params.control_strength,
        params.style_ratio,
        params.normalize_input,
        params.input_id_images_path.c_str()
    );
    bool ret = writeImg(params, params.batch_count, result);
    releaseImg(&input_image);
    releaseImg(&control_image);
    free(result);
    result = NULL;
    return ret;
}

static bool paintImg2Vid(SDParams& params, sd_ctx_t* sd_ctx) {
    sd_image_t* input_image { loadInputImage(params) };
    sd_image_t* result = img2vid(sd_ctx,
        *input_image,
        params.width,
        params.height,
        params.video_frames,
        params.motion_bucket_id,
        params.fps,
        params.augmentation_level,
        params.min_cfg,
        params.cfg_scale,
        params.sample_method,
        params.sample_steps,
        params.strength,
        params.seed
    );
    bool ret = writeImg(params, params.video_frames, result);
    releaseImg(&input_image);
    free(result);
    result = NULL;
    return ret;
}

static bool writeImg(SDParams& params, size_t count, sd_image_t* result) {
    if(result == NULL) {
        return false;
    }
    for (size_t i = 0; i < count; i++) {
        if (result[i].data == NULL) {
            continue;
        }
        std::string output_file = lifuren::files::join({params.output_path, std::to_string(lifuren::uuid()) + "_" + std::to_string(i) + ".png"}).string();
        SPDLOG_DEBUG("生成图片：{}", output_file);
        lifuren::images::write(output_file, result[i].data, params.width, params.height);
        free(result[i].data);
        result[i].data = NULL;
    }
    return true;
}

static void releaseImg(sd_image_t** image) {
    if(image == nullptr || *image == nullptr) {
        return;
    }
    delete[] (*image)->data;
    (*image)->data = nullptr;
    delete *image;
    *image = nullptr;
}

static sd_ctx_t* getSDCtx(SDParams& params, bool vae_decode_only) {
    // TODO: 线程安全
    if(shareSDCtx && share_sd_ctx != nullptr) {
        return share_sd_ctx;
    }
    sd_ctx_t* sd_ctx = new_sd_ctx(
        params.model_path.c_str(),
        params.clip_l_path.c_str(),
        params.t5xxl_path.c_str(),
        params.diffusion_model_path.c_str(),
        params.vae_path.c_str(),
        params.taesd_path.c_str(),
        params.controlnet_path.c_str(),
        params.lora_model_dir.c_str(),
        params.embeddings_path.c_str(),
        params.stacked_id_embeddings_path.c_str(),
        vae_decode_only,
        params.vae_tiling,
        true,
        params.n_threads,
        params.wtype,
        params.rng_type,
        params.schedule,
        params.clip_on_cpu,
        params.control_net_cpu,
        params.vae_on_cpu
    );
    if(shareSDCtx && sd_ctx != nullptr) {
        share_sd_ctx = sd_ctx;
    }
    return sd_ctx;
}
