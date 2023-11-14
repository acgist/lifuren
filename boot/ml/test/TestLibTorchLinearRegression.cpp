#include "../src/header/LibTorch.hpp"

int main(int argc, char const *argv[]) {
    lifuren::init(argc, argv);
    LOG(INFO) << "测试";
    lifuren::testLibTorchLinearRegression();
    LOG(INFO) << "完成";
    lifuren::shutdown();
    return 0;
}
