#include "lifuren/Wudaozi.hpp"

#include <cmath>

#include "spdlog/spdlog.h"

#include "opencv2/opencv.hpp"

#include "lifuren/File.hpp"
#include "lifuren/Layer.hpp"
#include "lifuren/Trainer.hpp"

#ifndef LFR_DROPOUT
#define LFR_DROPOUT 0.3
#endif

namespace lifuren {

/**
 * UNet
 */
class UNetImpl : public torch::nn::Module {

private:
    torch::nn::Conv2d stem{ nullptr };
    torch::nn::Sequential time_embed{ nullptr };
    torch::nn::ModuleDict encoder_blocks{nullptr};
    torch::nn::ModuleDict decoder_blocks{nullptr};

public:
    UNetImpl(int img_height, int img_width, int channels, int model_channels, int embedding_channels,  int min_pixel = 4,
        int n_block = 2, int n_groups = 32, int attn_resolution = 16, const std::vector<int>& scales = { 1, 1, 2, 2, 4, 4 }) {
        this->stem = this->register_module("stem", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, embedding_channels, { 3, 3 }).padding(1)));
        this->time_embed = this->register_module("time_embed", torch::nn::Sequential(
            torch::nn::Linear(model_channels, embedding_channels),
            torch::nn::SiLU(),
            torch::nn::Linear(embedding_channels, embedding_channels)
        ));
        int min_img_size = std::min(img_height, img_width);
        torch::OrderedDict<std::string, std::shared_ptr<Module>> encoder_blocks;
        std::vector<std::tuple<int, int>> encoder_channels;
        int cur_c = embedding_channels;
        auto skip_pooling = 0;
        for (size_t i = 0; i < scales.size(); i++) {
            auto scale = scales[i];
            // sevaral residual blocks
            for (size_t j = 0; j < n_block; j++) {
                encoder_channels.emplace_back(cur_c, scale * embedding_channels);
                auto block = lifuren::nn::ResidualBlock(cur_c, scale * embedding_channels, embedding_channels, n_groups);
                cur_c = scale * embedding_channels;
                encoder_blocks.insert((std::stringstream() << "enc_block_" << i * n_block + j).str(), block.ptr());
            }
    
            if (min_img_size <= attn_resolution) {
                encoder_blocks.insert((std::stringstream() << "attn_enc_block_" << i * n_block).str(),
                lifuren::nn::AttentionBlock(cur_c, 8, cur_c / 8).ptr());
            }
    
            // downsample block if not reach to `min_pixel`.
            if (min_img_size > min_pixel) {
                encoder_blocks.insert((std::stringstream() << "down_block_" << i).str(), lifuren::nn::Downsample(cur_c).ptr());
                min_img_size = min_img_size / 2;
            } else {
                skip_pooling += 1; // log how many times skip pooling.
            }
        }
        // mid
        encoder_blocks.insert((std::stringstream() << "enc_block_" << scales.size() * n_block).str(),
        lifuren::nn::ResidualBlock(cur_c, cur_c, embedding_channels, n_groups).ptr());
            // ?
            // lifuren::nn::ResidualBlock(ch, ch, time_embed_dim, dropout),
            // lifuren::nn::AttentionBlock(ch, num_heads=num_heads),
            // lifuren::nn::ResidualBlock(ch, ch, time_embed_dim, dropout)

        this->encoder_blocks = this->register_module("encoder_blocks", torch::nn::ModuleDict(encoder_blocks));

        std::reverse(encoder_channels.begin(), encoder_channels.end());

        torch::OrderedDict<std::string, std::shared_ptr<Module>> decoder_blocks;
        size_t m = 0;
        for (int i = scales.size() - 1; i > -1; i--) {
            auto rev_scale = scales[i];
            if (m >= skip_pooling) {
                decoder_blocks.insert((std::stringstream() << "up_block_" << m).str(), lifuren::nn::Upsample(cur_c).ptr());
                min_img_size *= 2;
            }

            for (size_t j = 0; j < n_block; j++) {
                int out_channels;
                int in_channels;
                std::tie(out_channels, in_channels) = encoder_channels[m * n_block + j];
                decoder_blocks.insert((std::stringstream() << "dec_block_" << m * n_block + j).str(),
                lifuren::nn::ResidualBlock(in_channels, out_channels, embedding_channels, n_groups).ptr());
                cur_c = out_channels;
            }

            if (min_img_size <= attn_resolution) {
                decoder_blocks.insert((std::stringstream() << "attn_dec_block_" << m * n_block).str(),
                lifuren::nn::AttentionBlock(cur_c, 8, cur_c / 8).ptr());
            }

            m++;
        }
        torch::nn::Sequential out(
            torch::nn::GroupNorm(n_groups, cur_c),
                torch::nn::SiLU(),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(cur_c, channels, {3, 3}).padding(1).bias(false))
        );
        decoder_blocks.insert(std::string("out"), out.ptr());
        this->decoder_blocks = this->register_module("decoder_blocks", torch::nn::ModuleDict(decoder_blocks));
    }
    ~UNetImpl() {
        this->unregister_module("stem");
        this->unregister_module("time_embed");
    }

public:
    torch::Tensor forward(torch::Tensor x, torch::Tensor t) {
        x = stem(x);
        t = this->time_embed->forward(t);

        std::vector<torch::Tensor> inners;
    
        inners.push_back(x);
        for (const auto &item: encoder_blocks->items()) {
            auto name = item.first;
            auto module = item.second;
            // resudial block
            if (name.starts_with("enc")) {
                x = module->as<lifuren::nn::ResidualBlock>()->forward(x, t);
                inners.push_back(x);
            } else if (name.starts_with("attn")) {
                x = module->as<lifuren::nn::AttentionBlock>()->forward(x);
            }
                // downsample block
            else {
                x = module->as<lifuren::nn::Downsample>()->forward(x);
                inners.push_back(x);
            }
        }
    
        // drop last two (contains middle block output)
        auto inners_ = std::vector<torch::Tensor>(inners.begin(), inners.end() - 2);
    
        for (const auto &item: decoder_blocks->items()) {
            auto name = item.first;
            auto module = item.second;
    
            // upsample block
            if (name.starts_with("up")) {
                x = module->as<lifuren::nn::Upsample>()->forward(x);
                torch::Tensor xi = inners_.back();
                inners_.pop_back(); // pop()
                x = x + xi;
            }
                // resudial block
            else if (name.starts_with("dec")) {
                torch::Tensor xi = inners_.back();
                inners_.pop_back(); // pop()
                x = module->as<lifuren::nn::ResidualBlock>()->forward(x, t);
                x = x + xi;
            } else if (name.starts_with("attn")) {
    
                x = module->as<lifuren::nn::AttentionBlock>()->forward(x);
            } else {
                x = module->as<torch::nn::Sequential>()->forward(x);
            }
        }
    
        return x;
    }

};

TORCH_MODULE(UNet);

/**
 * 变形
 */
class Reshape : public torch::nn::Module {

private:
    std::vector<int64_t> shape;

public:
    Reshape(std::vector<int64_t> shape) : shape(shape) {
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return torch::reshape(input, this->shape);
    }

};

/**
 * 2D编码器
 */
class Encoder2dImpl : public torch::nn::Module {
    
private:
    torch::nn::Sequential encoder_2d{ nullptr };

public:
    Encoder2dImpl(int in, int out, bool output = false) {
        torch::nn::Sequential encoder_2d;
        // flatten
        encoder_2d->push_back(torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1).end_dim(2)));
        // conv
        encoder_2d->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).stride(1).padding(1)));
        encoder_2d->push_back(torch::nn::BatchNorm2d(out));
        encoder_2d->push_back(torch::nn::Tanh());
        encoder_2d->push_back(torch::nn::MaxPool2d(2));
        // conv
        encoder_2d->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3).stride(1).padding(1)));
        encoder_2d->push_back(torch::nn::BatchNorm2d(out));
        encoder_2d->push_back(torch::nn::Tanh());
        encoder_2d->push_back(torch::nn::MaxPool2d(2));
        // conv
        encoder_2d->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3).stride(1).padding(1)));
        encoder_2d->push_back(torch::nn::MaxPool2d(2));
        this->encoder_2d = this->register_module("encoder_2d", encoder_2d);
    }
    ~Encoder2dImpl() {
        this->unregister_module("encoder_2d");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return this->encoder_2d->forward(input);
    }

};

TORCH_MODULE(Encoder2d);

/**
 * 3D编码器
 */
class Encoder3dImpl : public torch::nn::Module {
    
private:
    int channel;
    torch::nn::Sequential encoder_3d{ nullptr };

public:
    Encoder3dImpl(int h_3d, int w_3d, int channel) : channel(channel) {
        torch::nn::Sequential encoder_3d;
        // conv
        encoder_3d->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(channel - 1, channel - 1, 3).stride(1).padding(1)));
        encoder_3d->push_back(torch::nn::BatchNorm3d(channel - 1));
        encoder_3d->push_back(torch::nn::Tanh());
        encoder_3d->push_back(torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions({ 2, 2, 2 }).stride({ 1, 2, 2 })));
        // conv
        encoder_3d->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(channel - 1, channel - 1, 3).stride(1).padding(1)));
        encoder_3d->push_back(torch::nn::BatchNorm3d(channel - 1));
        encoder_3d->push_back(torch::nn::Tanh());
        encoder_3d->push_back(torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions({ 2, 2, 2 }).stride({ 1, 2, 2 })));
        // conv
        encoder_3d->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(channel - 1, channel - 1, { 1, 3, 3 }).stride(1).padding({ 0, 1, 1 })));
        encoder_3d->push_back(torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions({ 1, 2, 2 }).stride({ 1, 2, 2 })));
        // out
        encoder_3d->push_back(torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(2).end_dim(4)));
        encoder_3d->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({ h_3d * w_3d })));
        this->encoder_3d = this->register_module("encoder_3d", encoder_3d);

    }
    ~Encoder3dImpl() {
        this->unregister_module("encoder_3d");
    }

public:
    torch::Tensor forward(torch::Tensor input) {
        return this->encoder_3d->forward(input.slice(1, 1, this->channel) - input.slice(1, 0, this->channel - 1));
    }

};

TORCH_MODULE(Encoder3d);

/**
 * 解码器
 */
class DecoderImpl : public torch::nn::Module {

private:
    torch::Tensor         encoder_hid{ nullptr };
    torch::nn::GRU        encoder_gru{ nullptr };
    torch::nn::Sequential decoder_3d { nullptr };

public:
    DecoderImpl(int h, int w, int scale, int batch, int channel, int num_layers = 3) {
        int w_3d = w / scale;
        int h_3d = h / scale;
        // 3D编码器
        torch::nn::Sequential decoder_3d;
        decoder_3d->push_back(lifuren::Reshape({ batch, channel - 1, h_3d, w_3d }));
        decoder_3d->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(channel - 1, 1, 3).stride(1).padding(1)));
        decoder_3d->push_back(lifuren::Reshape({ batch, 1, h_3d * w_3d }));
        decoder_3d->push_back(torch::nn::Dropout(LFR_DROPOUT));
        decoder_3d->push_back(torch::nn::Linear(h_3d * w_3d, h * w));
        decoder_3d->push_back(lifuren::Reshape({ batch, 1, h, w }));
        decoder_3d->push_back(torch::nn::Tanh());
        this->decoder_3d = this->register_module("decoder_3d", decoder_3d);
        // GRU
        this->encoder_hid = torch::zeros({ num_layers, batch, h_3d * w_3d }).to(LFR_DTYPE).to(lifuren::get_device());
        this->encoder_gru = this->register_module("encoder_gru", torch::nn::GRU(torch::nn::GRUOptions(h_3d * w_3d, h_3d * w_3d).num_layers(num_layers).batch_first(true).dropout(num_layers == 1 ? 0.0 : LFR_DROPOUT)));
    }
    ~DecoderImpl() {
        this->unregister_module("decoder_3d");
        this->unregister_module("encoder_gru");
    }

public:
    torch::Tensor forward(torch::Tensor input_2d, torch::Tensor input_3d) {
        auto [ o_o, o_h ] = this->encoder_gru->forward(input_3d, this->encoder_hid);
        return this->decoder_3d->forward(o_o);
    }

};

TORCH_MODULE(Decoder);

/**
 * 吴道子模型（视频生成）
 */
class WudaoziImpl : public torch::nn::Module {

private:
    lifuren::config::ModelParams params;
    int timesteps = 1000;
    UNet unet { nullptr };
    Encoder2d encoder_2d_1{ nullptr };
    Encoder3d encoder_3d_1{ nullptr };
    Decoder   decoder_1   { nullptr };
    
public:
    WudaoziImpl(lifuren::config::ModelParams params = {}) : params(params) {
        const int scale = 8;
        const int batch_size = static_cast<int>(this->params.batch_size);
        // this->encoder_2d_1 = this->register_module("encoder_2d_1", lifuren::Encoder2d(3 * LFR_VIDEO_QUEUE_SIZE, 16));
        // this->encoder_3d_1 = this->register_module("encoder_3d_1", lifuren::Encoder3d(LFR_IMAGE_HEIGHT / scale, LFR_IMAGE_WIDTH / scale, LFR_VIDEO_QUEUE_SIZE));
        // this->decoder_1    = this->register_module("decoder_1",    lifuren::Decoder(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, scale, batch_size, LFR_VIDEO_QUEUE_SIZE));
        this->unet = this->register_module("unet", UNet(LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH, 3, 20, 64));
    }
    ~WudaoziImpl() {
        // this->unregister_module("encoder_2d_1");
        // this->unregister_module("encoder_3d_1");
        // this->unregister_module("decoder_1");
    }

public:
    torch::Tensor forward(torch::Tensor feature) {
        return this->unet->forward(feature, torch::ones({20}));
    }

};

TORCH_MODULE(Wudaozi);

/**
 * 吴道子模型训练器（视频生成）
 */
class WudaoziTrainer : public lifuren::Trainer<torch::optim::Adam, lifuren::Wudaozi, lifuren::dataset::SeqDatasetLoader> {

public:
    WudaoziTrainer(lifuren::config::ModelParams params = {}) : Trainer(params) {
    }
    ~WudaoziTrainer() {
    }
    
public:
    void defineDataset() override {
        if(lifuren::file::exists(this->params.train_path)) {
            this->trainDataset = lifuren::dataset::image::loadWudaoziDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.train_path);
        }
        if(lifuren::file::exists(this->params.val_path)) {
            this->valDataset = lifuren::dataset::image::loadWudaoziDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.val_path);
        }
        if(lifuren::file::exists(this->params.test_path)) {
            this->testDataset = lifuren::dataset::image::loadWudaoziDatasetLoader(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT, this->params.batch_size, this->params.test_path);
        }
    }
    void defineOptimizer() override {
        torch::optim::AdamOptions optims;
        optims.lr(this->params.lr);
        // optims.eps(0.0001);
        this->optimizer = std::make_unique<torch::optim::Adam>(this->model->parameters(), optims);
    }
    void loss(torch::Tensor& feature, torch::Tensor& label, torch::Tensor& pred, torch::Tensor& loss) override {
        // L1Loss
        // MSELoss
        // HuberLoss
        // SmoothL1Loss
        // CrossEntropyLoss
        // auto noise = torch::randn_like(feature).to(lifuren::get_device());
        // auto d_noise = this->model->forward(noise, feature);
        auto pred = this->model->forward(label);
        loss = torch::mse_loss(pred, label);
    }
    torch::Tensor pred(torch::Tensor feature) {
        torch::NoGradGuard no_grad;
        // return this->model->forward(feature, 1000);
        return this->model->forward(feature);
    }

};

template<typename T>
class WudaoziClientImpl : public ClientImpl<lifuren::config::ModelParams, std::string, std::string, T> {

public:
    std::tuple<bool, std::string> pred(const std::string& input) override;
    std::tuple<bool, std::string> predImage(const std::string& input);
    std::tuple<bool, std::string> predVideo(const std::string& input);

};

};

template<>
std::tuple<bool, std::string> lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>::predImage(const std::string& input) {
    auto image = cv::imread(input);
    if(image.empty()) {
        return { false, {} };
    }
    const auto output = lifuren::file::modify_filename(input, ".mp4", "gen");
    cv::VideoWriter writer(output, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), LFR_VIDEO_FPS, cv::Size(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT));
    if(!writer.isOpened()) {
        SPDLOG_WARN("视频文件打开失败：{}", output);
        return { false, output };
    }
    int index = 0;
    lifuren::dataset::image::resize(image, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
    auto tensor = lifuren::dataset::image::mat_to_tensor(image).to(LFR_DTYPE).to(lifuren::get_device());
    std::vector<torch::Tensor> images;
    torch::Tensor bos = torch::zeros({ 3, LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH }).to(LFR_DTYPE).to(lifuren::get_device());
    torch::Tensor pad = torch::ones ({ 3, LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH }).to(LFR_DTYPE).to(lifuren::get_device());
    images.push_back(bos);
    for(int i = 0; i < LFR_VIDEO_QUEUE_SIZE - 2; ++i) {
        images.push_back(pad);
    }
    images.push_back(tensor);
    for(int i = 0; i < LFR_VIDEO_FRAME_SIZE; ++i) {
        auto result = this->trainer->pred(torch::stack(images, 0).unsqueeze(0)).squeeze(0);
        lifuren::dataset::image::tensor_to_mat(image, result.to(torch::kFloat32).to(torch::kCPU));
        writer.write(image);
        images.erase(images.begin());
        images.push_back(result);
    }
    writer.release();
    return { true, output };
}

template<>
std::tuple<bool, std::string> lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>::predVideo(const std::string& input) {
    cv::VideoCapture video(input);
    if(!video.isOpened()) {
        return { false, {} };
    }
    const auto output = lifuren::file::modify_filename(input, ".mp4", "gen");
    cv::VideoWriter writer(output, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), LFR_VIDEO_FPS, cv::Size(LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT));
    if(!writer.isOpened()) {
        SPDLOG_WARN("视频文件打开失败：{}", output);
        return { false, output };
    }
    int index = 0;
    cv::Mat image;
    std::vector<torch::Tensor> images;
    torch::Tensor bos = torch::zeros({ 3, LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH }).to(LFR_DTYPE).to(lifuren::get_device());
    torch::Tensor pad = torch::ones ({ 3, LFR_IMAGE_HEIGHT, LFR_IMAGE_WIDTH }).to(LFR_DTYPE).to(lifuren::get_device());
    images.push_back(bos);
    for(int i = 0; i < LFR_VIDEO_QUEUE_SIZE - 2; ++i) {
        images.push_back(pad);
    }
    while(video.read(image)) {
        lifuren::dataset::image::resize(image, LFR_IMAGE_WIDTH, LFR_IMAGE_HEIGHT);
        auto tensor = lifuren::dataset::image::mat_to_tensor(image).to(LFR_DTYPE).to(lifuren::get_device());
        images.push_back(tensor);
        auto result = this->trainer->pred(torch::stack(images, 0).unsqueeze(0)).squeeze(0);
        lifuren::dataset::image::tensor_to_mat(image, result.to(torch::kFloat32).to(torch::kCPU));
        writer.write(image);
        images.erase(images.begin());
    }
    video.release();
    writer.release();
    return { true, output };
}

template<>
std::tuple<bool, std::string> lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>::pred(const std::string& input) {
    if(!this->trainer) {
        return { false, {} };
    }
    auto suffix = lifuren::file::file_suffix(input);
    if(suffix == ".mp4") {
        return this->predVideo(input);
    } else {
        return this->predImage(input);
    }
}

std::unique_ptr<lifuren::WudaoziClient> lifuren::get_wudaozi_client() {
    return std::make_unique<lifuren::WudaoziClientImpl<lifuren::WudaoziTrainer>>();
}
