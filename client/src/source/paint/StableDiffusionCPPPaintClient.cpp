/**
 * https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/cli/main.cpp
 */
#include "lifuren/Client.hpp"

#include "spdlog/spdlog.h"

#include "stable-diffusion.h"

const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
};

const char* sample_method_str[] = {
    "euler_a",
    "euler",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "lcm",
};

const char* schedule_str[] = {
    "default",
    "discrete",
    "karras",
    "ays",
};

const char* modes_str[] = {
    "txt2img",
    "img2img",
    "img2vid",
    "convert",
};

enum class SDMode {
    TXT2IMG,
    IMG2IMG,
    IMG2VID,
    CONVERT,
    MODE_COUNT
};

struct SDParams {
    int n_threads = -1;
    SDMode mode   = SDMode::TXT2IMG;
    std::string model_path;
    std::string clip_l_path;
    std::string t5xxl_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string esrgan_path;
    std::string controlnet_path;
    std::string embeddings_path;
    std::string stacked_id_embeddings_path;
    std::string input_id_images_path;
    sd_type_t   wtype = SD_TYPE_COUNT;
    std::string lora_model_dir;
    std::string output_path = "output.png";
    std::string input_path;
    std::string control_image_path;
    std::string prompt;
    std::string negative_prompt;

    float min_cfg     = 1.0f;
    float cfg_scale   = 7.0f;
    float guidance    = 3.5f;
    float style_ratio = 20.f;
    int clip_skip     = -1;  // <= 0 represents unspecified
    int width         = 512;
    int height        = 512;
    int batch_count   = 1;

    int video_frames         = 6;
    int motion_bucket_id     = 127;
    int fps                  = 6;
    float augmentation_level = 0.f;

    sample_method_t sample_method = EULER_A;
    schedule_t schedule           = DEFAULT;
    int sample_steps              = 20;
    float strength                = 0.75f;
    float control_strength        = 0.9f;
    rng_type_t rng_type           = CUDA_RNG;
    int64_t seed                  = 42;
    bool verbose                  = false;
    bool vae_tiling               = false;
    bool control_net_cpu          = false;
    bool normalize_input          = false;
    bool clip_on_cpu              = false;
    bool vae_on_cpu               = false;
    bool canny_preprocess         = false;
    bool color                    = false;
    int upscale_repeats           = 1;
};

lifuren::StableDiffusionCPPPaintClient::StableDiffusionCPPPaintClient() {
}

lifuren::StableDiffusionCPPPaintClient::~StableDiffusionCPPPaintClient() {
}

bool lifuren::StableDiffusionCPPPaintClient::paint(const std::string& prompt, lifuren::PaintClient::PaintCallback callback, const std::string& image) {
    if(this->commandClient) {
        SPDLOG_WARN("加载StableDiffusionCPP任务失败：已经存在任务");
        return false;
    }
    this->commandClient = std::make_unique<lifuren::CommandClient>("");
    return true;
}
