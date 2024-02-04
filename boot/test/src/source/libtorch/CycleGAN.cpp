#include "../../header/GAN.hpp"

// https://blog.csdn.net/jizhidexiaoming/article/details/128619117
// // Down sampling : 通过conv2d进行两次下采样，同时double channels
// class DownSampleImpl : public torch::nn::Module {
// public:
//     DownSampleImpl(int in_channels, int out_channels);
//     torch::Tensor forward(torch::Tensor& x);
// private:
//     torch::nn::Conv2d conv1{ nullptr };
//     torch::nn::InstanceNorm2d bn1{ nullptr };
//     torch::nn::ReLU relu1{ nullptr };
// };

// TORCH_MODULE(DownSample);

// DownSampleImpl::DownSampleImpl(int in_channels, int out_channels) {
//     /**
//      * 卷积
//      * stride：步长
//      * dilation：空洞卷积
//      * kernel_size：卷积核大小
//      * https://ezyang.github.io/convolution-visualizer/index.html
//      * output_sie = (input_size - kernel_size + 2 * padding) + 1
//      */
//     this->conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(2).padding(1));
//     /**
//      * 归一化
//      * https://blog.csdn.net/qq_43665602/article/details/126551756
//      * InstanceNorm1d
//      * InstanceNorm2d
//      * BatchNorm1d
//      * BatchNorm2d
//      */
//     this->bn1 = torch::nn::InstanceNorm2d(out_channels);
//     /**
//      * 激活函数
//      * tanh
//      * ReLU
// 	 * sigmoid
//      * softmax
//      * Leaky ReLU
//      */
//     this->relu1 = torch::nn::ReLU(true);
//     this->register_module("generator downsample pad1",  this->conv1);
//     this->register_module("generator downsample bn1",   this->bn1);
//     this->register_module("generator downsample relu1", this->relu1);
// }

// torch::Tensor DownSampleImpl::forward(torch::Tensor& x) {
//     x = this->conv1(x);
//     x = this->bn1(x);
//     x = this->relu1(x);
//     return x;
// }

// // two conv2d+bn+relu. keep feature scale.
// class ResidualBlockImpl : public torch::nn::Module {
// public:
//     ResidualBlockImpl(int in_channels);
//     torch::Tensor forward(torch::Tensor x);
// private:
//     torch::nn::ReflectionPad2d pad1{ nullptr };
//     torch::nn::Conv2d conv1{ nullptr };
//     torch::nn::InstanceNorm2d bn1{ nullptr };
//     torch::nn::ReLU relu1{ nullptr };
//     torch::nn::ReflectionPad2d pad2{ nullptr };
//     torch::nn::Conv2d conv2{ nullptr };
//     torch::nn::InstanceNorm2d bn2{ nullptr };
// };

// TORCH_MODULE(ResidualBlock);

// ResidualBlockImpl::ResidualBlockImpl(int in_channels) {
//     this->pad1  = torch::nn::ReflectionPad2d(1);
//     this->conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3));
//     this->bn1   = torch::nn::InstanceNorm2d(in_channels);
//     this->relu1 = torch::nn::ReLU(true);
//     this->pad2  = torch::nn::ReflectionPad2d(1);
//     this->conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3));
//     this->bn2   = torch::nn::InstanceNorm2d(in_channels);
//     this->register_module("block pad1",  this->pad1);
//     this->register_module("block conv1", this->conv1);
//     this->register_module("block bn1",   this->bn1);
//     this->register_module("block pad2",  this->pad2);
//     this->register_module("block conv2", this->conv2);
//     this->register_module("block bn2",   this->bn2);
// }

// torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
//     x = this->pad1(x);
//     x = this->conv1(x);
//     x = this->bn1(x);
//     x = this->relu1(x);
//     x = this->pad2(x);
//     x = this->conv2(x);
//     x = this->bn2(x);
//     return x;
// }

// /// 两次上采样，(b, 256, 64, 64) -> (b, 128, 128, 128) -> (b, 64, 256, 256)
// class UpSampleBlockImpl : public torch::nn::Module {
// public:
//     UpSampleBlockImpl(int in_channels, int out_channels);
//     torch::Tensor forward(torch::Tensor x);
// private:
//     torch::nn::Upsample up{ nullptr };
//     torch::nn::Conv2d conv{ nullptr };
//     torch::nn::InstanceNorm2d bn{ nullptr };
//     torch::nn::ReLU relu{ nullptr };
// };

// TORCH_MODULE(UpSampleBlock);

// UpSampleBlockImpl::UpSampleBlockImpl(int in_channels, int out_channels) {
//     this->up   = torch::nn::Upsample(upsample_options(std::vector<double>({2, 2})));
//     this->conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1));
//     this->bn   = torch::nn::InstanceNorm2d(out_channels);
//     this->relu = torch::nn::ReLU(true);
//     this->register_module("generator UpSampleBlock upsample", this->up);
//     this->register_module("generator UpSampleBlock conv",     this->conv);
//     this->register_module("generator UpSampleBlock bn",       this->bn);
//     this->register_module("generator UpSampleBlock relu",     this->relu);
// }

// torch::Tensor UpSampleBlockImpl::forward(torch::Tensor x) {
//     x = this->up(x);
//     x = this->conv(x);
//     x = this->bn(x);
//     x = this->relu(x);
//     return x;
// }

// /// 下采样，res_blocks，上采样，output layer.
// class GeneratorResNetImpl : public torch::nn::Module {
// public:
// 	GeneratorResNetImpl(std::vector<int> input_shape, int num_residual_blocks);
// 	torch::Tensor forward(torch::Tensor x);
// private:
// 	torch::nn::Sequential _make_layer(int in_channels, int blocks);
// 	torch::nn::ReflectionPad2d pad1{ nullptr };
// 	torch::nn::Conv2d conv1{ nullptr };
// 	torch::nn::InstanceNorm2d bn1{ nullptr };
// 	torch::nn::ReLU relu1{ nullptr };
// 	// down
// 	DownSample down1{ nullptr };
// 	DownSample down2{ nullptr };
// 	// res
// 	torch::nn::Sequential res_blocks = torch::nn::Sequential();
// 	// up
// 	UpSampleBlock up1{ nullptr };
// 	UpSampleBlock up2{ nullptr };
// 	// output layer
// 	torch::nn::ReflectionPad2d pad2{ nullptr };
// 	torch::nn::Conv2d conv2{ nullptr };
// 	torch::nn::Tanh tanh2{ nullptr };
// };

// TORCH_MODULE(GeneratorResNet);

// torch::nn::Sequential GeneratorResNetImpl::_make_layer(int in_channels, int blocks) {
// 	torch::nn::Sequential layers;
// 	for (int i = 0; i < blocks; i++) {
// 		layers->push_back(ResidualBlock(in_channels));
// 	}
// 	return layers;
// }

// GeneratorResNetImpl::GeneratorResNetImpl(std::vector<int> input_shape, int num_residual_blocks) {
// 	int channels = input_shape[0]; // 3
// 	int out_channels = 64;
// 	// 1, conv + bn + relu. (256 + 6 - 7 + 2 * 0) / 1 + 1 = 256
// 	this->pad1  = torch::nn::ReflectionPad2d(channels);
// 	this->conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, out_channels, 7));
// 	this->bn1   = torch::nn::InstanceNorm2d(out_channels);
// 	this->relu1 = torch::nn::ReLU(true);
// 	int in_channels = out_channels;
// 	// 2, Down sampling: 通过conv2d两次下采样，并且double channels
// 	this->down1 = DownSample(in_channels, out_channels * 2);
// 	this->down2 = DownSample(out_channels * 2, out_channels * 4);
// 	in_channels = out_channels * 4; // 256 = 64 * 4
// 	// 3, Residual blocks: keep feature scale and channel unchange.
// 	this->res_blocks = this->_make_layer(in_channels, num_residual_blocks); // (b, 256, 64, 64)
// 	// 4, Up sampling: up+conv+bn+relu. halve channels and keep feature scale unchange.
// 	this->up1 = UpSampleBlock(in_channels, in_channels / 2); // (b, 128, 128, 128)
// 	this->up2 = UpSampleBlock(in_channels / 2, in_channels / 4); // (b, 64, 256, 256)
// 	in_channels = in_channels / 4; // 64
// 	// 5, output layer: pad+conv+tanh. change channels and keep feature scale unchange.
// 	this->pad2  = torch::nn::ReflectionPad2d(channels); // 3
// 	this->conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, channels, 7)); // (b, 64, 256, 256) -> (b, 3, 256, 256)
// 	this->tanh2 = torch::nn::Tanh();
// 	this->register_module("generator pad1",       this->pad1);
// 	this->register_module("generator conv1",      this->conv1); // 一定要注册，不然不会使用cuda
// 	this->register_module("generator bn1",        this->bn1);
// 	this->register_module("generator relu1",      this->relu1);
// 	this->register_module("generator down1",      this->down1);
// 	this->register_module("generator down2",      this->down2);
// 	this->register_module("generator res_blocks", this->res_blocks);
// 	this->register_module("generator up1",        this->up1);
// 	this->register_module("generator up2",        this->up2);
// 	this->register_module("generator pad2",       this->pad2);
// 	this->register_module("generator conv2",      this->conv2);
// 	this->register_module("generator tanh2",      this->tanh2);
// }

// torch::Tensor GeneratorResNetImpl::forward(torch::Tensor x) { // (b, 3, 256, 256)
// 	// 1, conv + bn + relu. (256 + 6 - 7 + 2 * 0) / 1 + 1 = 256
// 	x = this->pad1(x);
// 	x = this->conv1(x);
// 	x = this->bn1(x);
// 	x = this->relu1(x); // (b, 64, 256, 256)
// 	// 2, Down sampling: 通过conv2d两次下采样，并且double channels
// 	x = this->down1(x); // (b, 128, 128, 128)
// 	x = this->down2(x); // (b, 256, 64, 64)
// 	// 3, Residual blocks: keep feature scale and channel unchange.
// 	x = this->res_blocks->forward(x); // (b, 256, 64, 64)
// 	// 4, Up sampling: up+conv+bn+relu. halve channels and keep feature scale unchange.
// 	x = this->up1(x); // (b, 128, 128, 128)
// 	x = this->up2(x); // (b, 64, 256, 256)
// 	// 5, output layer: pad+conv+tanh. change channels and keep feature scale unchange.
// 	x = this->pad2(x);
// 	x = this->conv2(x);
// 	x = this->tanh2(x); // (b, 3, 256, 256)
// 	std::cout << x.sizes() << std::endl;
// 	return x;
// }

void lifuren::testCycleGAN() {
}
