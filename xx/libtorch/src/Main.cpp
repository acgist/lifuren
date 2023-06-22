#include <iostream>
#include "torch/script.h"
// #include "torch/library.h"
#include "header/LibTorch.hpp"

int main(int argc, char const *argv[]) {
    lifuren::initGlog(argc, argv);
    LOG(INFO) << "测试";
    lifuren::shutdownGlog();
    // torch::Tensor output = torch::randn({ 3, 2 });
    // std::cout << output;
    return 0;
}
