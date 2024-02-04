#include "GLog.hpp"
#include "torch/torch.h"

int main(const int argc, const char * const argv[]) {
    lifuren::init(argc, argv);
    torch::nn::Linear fc1(50, 10);
    torch::nn::Linear fc2 = torch::nn::Linear(50, 10);
    LOG(INFO) << sizeof(fc1) << "\n";
    LOG(INFO) << sizeof(fc2) << "\n";
    return 0;
}
