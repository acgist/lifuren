
#include "../../header/LibTorch.hpp"

#include "Logger.hpp"

#include "torch/torch.h"
#include "torch/script.h"

#include "spdlog/spdlog.h"

#include "spdlog/fmt/ostr.h"
#include "spdlog/fmt/ranges.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

LFR_LOG_FORMAT_STREAM(at::Tensor);
LFR_LOG_FORMAT_STREAM(torch::jit::IValue);

static void testDeeplabLocal() {
    auto model = torch::jit::load("D:\\download\\deeplabv3_resnet50.COCO_WITH_VOC_LABELS_V1.pt", torch::kCPU);
    model.to(torch::kCPU);
    model.eval();
    auto bike = cv::imread("D://download/yang.png");
    // auto bike = cv::imread("D://download/bike.jpg");
    // auto bike = cv::imread("D://download/bike2.webp");
    if(bike.empty()) {
        return;
    }
    cv::resize(bike, bike, cv::Size(512, 512), cv::INTER_LINEAR);
    cv::imshow("bike", bike);
    cv::waitKey(0);
    auto input_tensor = torch::from_blob(bike.data, { bike.rows, bike.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kFloat32) / 225.0;
    std::vector<float> m = { 0.485, 0.456, 0.406 };
    std::vector<float> v = { 0.229, 0.224, 0.225 };
    auto mean = torch::from_blob(m.data(), { 1, 3, 1, 1 }, torch::kFloat32);
    auto var  = torch::from_blob(v.data(), { 1, 3, 1, 1 }, torch::kFloat32);
    input_tensor = (input_tensor - mean) / var;
    std::vector<torch::jit::IValue> vector;
    vector.push_back(input_tensor);
    auto result = model.forward(vector).toGenericDict();
    SPDLOG_DEBUG("执行结果：{}", result.size());
    for(
        auto iterator = result.begin();
        iterator != result.end();
        ++iterator
    ) {
        auto resultImage = iterator->value().toTensor().mul(255).to(torch::kU8);
        cv::Mat image(cv::Size(512, 512), CV_8U, resultImage.data_ptr());
        SPDLOG_DEBUG("抠图：{}", iterator->key());
        cv::imshow("bike-result", image);
        cv::waitKey(0);
        image.release();
    }
}

static void testMobilenetLocal() {
    auto model = torch::jit::load("D:\\download\\model\\trace\\mobilenet_v3_large.IMAGENET1K_V2.pt", torch::kCPU);
    model.to(torch::kCPU);
    model.eval();
    std::vector<torch::jit::IValue> vector;
    vector.push_back(torch::ones({1, 3, 224, 224}));
    SPDLOG_DEBUG("执行结果：{}", vector);
    auto result = model.forward(vector).toTensor();
    SPDLOG_DEBUG("执行结果：{}", result);
    SPDLOG_DEBUG("执行结果：{}", result.max());
    SPDLOG_DEBUG("执行结果：{}", result.argmax());
    SPDLOG_DEBUG("执行结果：{}", result.slice(1, 0, 5));
}

void lifuren::testModel() {
    testDeeplabLocal();
    testMobilenetLocal();
}

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    try {
        lifuren::testModel();
    } catch(const std::exception& e) {
        SPDLOG_ERROR("加载模型异常：{}", e.what());
    }
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
