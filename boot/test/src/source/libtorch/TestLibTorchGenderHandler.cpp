#include "../../header/LibTorch.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    lifuren::Gender gender;
    lifuren::GenderHandler handler;
    handler.model = gender;
    if(torch::cuda::is_available()) {
        handler.device = torch::Device(torch::kCUDA);
    }
    #ifdef _WIN32
    handler.trainAndVal(
        32,
        8,
        0.001,
        "D:\\tmp\\gender",
        ".jpg",
        "D:\\tmp\\gender\\model.pt"
    );
    // handler.load("D:\\tmp\\gender\\model.pt");
    cv::Mat image = cv::imread("D:\\tmp\\yusheng.jpg");
    SPDLOG_DEBUG("预测结果：{}", handler.pred(image));
    image.release();
    image = cv::imread("D:\\tmp\\girl.png");
    SPDLOG_DEBUG("预测结果：{}", handler.pred(image));
    image.release();
    #else
    handler.trainAndVal(
        32,
        8,
        0.001,
        "/tmp/gender",
        ".jpg",
        "/tmp/gender/model.pt"
    );
    #endif
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
