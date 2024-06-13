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

static void testInit();
static void testClone();
static void testResize();
static void testSlice();
static void testOperator();
static void testSize();
static void testArgmax();
static void testSqueeze();
static void testPermute();
static void testImage();
static void testFind();
static void testTransfor();
static void testEqual();
static void testCUDA();
static void testGrad();

void lifuren::testTensor() {
    SPDLOG_DEBUG("是否支持CUDA：{}", torch::cuda::is_available());
    SPDLOG_DEBUG("是否支持CUDNN：{}", torch::cuda::cudnn_is_available());
    testInit();
    testClone();
    testResize();
    testSize();
    testArgmax();
    testSqueeze();
    testPermute();
    testImage();
    testFind();
    testTransfor();
    testEqual();
    testCUDA();
    testGrad();
}

static void testInit() {
    auto a = torch::zeros({3, 4});
    SPDLOG_DEBUG("a =\r\n{}", a);
    a = torch::ones({3, 4});
    SPDLOG_DEBUG("a =\r\n{}", a);
    a = torch::eye(4);
    SPDLOG_DEBUG("a =\r\n{}", a);
    a = torch::full({3, 4}, 10);
    SPDLOG_DEBUG("a =\r\n{}", a);
    a = torch::tensor({33, 22, 11});
    SPDLOG_DEBUG("a =\r\n{}", a);
    // 随机
    a = torch::rand({ 3, 4 });
    SPDLOG_DEBUG("a =\r\n{}", a);
    // 正态分布随机
    a = torch::randn({ 3, 4 });
    SPDLOG_DEBUG("a =\r\n{}", a);
    a = torch::randint(0, 4, { 3, 4 });
    SPDLOG_DEBUG("a =\r\n{}", a);
    int array[10] = { 3, 4, 5, 1, 2, 3 };
    a = torch::from_blob(array, { 3, 2 }, torch::kFloat);
    SPDLOG_DEBUG("a =\r\n{}", a);
    std::vector<float> vector{ 3, 4, 5, 1, 2, 3 };
    a = torch::from_blob(vector.data(), { 3, 2 }, torch::kFloat);
    SPDLOG_DEBUG("a =\r\n{}", a);
}

static void testClone() {
    auto a = torch::zeros({3, 4});
    // 浅拷贝
    auto b = a;
    SPDLOG_DEBUG("a =\r\n{}", a);
    SPDLOG_DEBUG("b =\r\n{}", b);
    a[0][0] = 1;
    SPDLOG_DEBUG("a =\r\n{}", a);
    SPDLOG_DEBUG("b =\r\n{}", b);
    // 深拷贝
    auto c = a.clone();
    SPDLOG_DEBUG("c =\r\n{}", a);
    SPDLOG_DEBUG("c =\r\n{}", c);
    a[0][0] = 2;
    SPDLOG_DEBUG("a =\r\n{}", a);
    SPDLOG_DEBUG("c =\r\n{}", c);
    // 已有尺寸建立新的张量
    // auto d = torch::zeros_like(b);
    // auto d = torch::ones_like(b);
    // auto d = torch::rand_like(b, torch::kFloat);
}

static void testResize() {
    // TODO: flatten、view、reshape、transpose
    int array[6] = { 4, 5, 6, 1, 2, 3 };
    auto a = torch::from_blob(array, { 3, 2 }, torch::kInt);
    SPDLOG_DEBUG("a =\r\n{}", a);
    auto b = a.view({ 2, 3 });
    // auto b = a.view({ 1, 2, -1 });
    SPDLOG_DEBUG("b =\r\n{}", b);
    SPDLOG_DEBUG("a =\r\n{}", a);
    SPDLOG_DEBUG("a =\r\n{}", a.sizes());
    SPDLOG_DEBUG("a =\r\n{}", a[0]);
    SPDLOG_DEBUG("a =\r\n{}", a[0][0]);
}

static void testSlice() {
    // TODO: narrow、select、index、index_put_、index_select、slice
}

static void testOperator() {
    int arrayA[] = { 1, 2, 3, 4 };
    int arrayB[] = { 1, 2, 3, 4 };
    const torch::Tensor lineA = torch::tensor({ 1, 2, 3, 4});
    const torch::Tensor lineB = torch::tensor({ 1, 2, 3, 4});
    const torch::Tensor a = torch::from_blob(arrayA, { 2, 2 }, torch::kInt);
    const torch::Tensor b = torch::from_blob(arrayB, { 2, 2 }, torch::kInt);
    SPDLOG_DEBUG("a =\r\n{}", a);
    SPDLOG_DEBUG("b =\r\n{}", b);
    SPDLOG_DEBUG("a + b  =\r\n{}", (a + b));
    SPDLOG_DEBUG("a - b  =\r\n{}", (a - b));
    SPDLOG_DEBUG("a * b  =\r\n{}", (a * b));
    SPDLOG_DEBUG("a / b  =\r\n{}", (a / b));
    SPDLOG_DEBUG("a % b  =\r\n{}", (a % b));
    SPDLOG_DEBUG("a == b =\r\n{}", (a == b));
    SPDLOG_DEBUG("lineA dot lineB   =\r\n{}", lineA.dot(lineB));
    SPDLOG_DEBUG("lineA dot lineB.t =\r\n{}", lineA.dot(lineB.t()));
    // TODO: cat、stack
}

static void testSize() {
    int array[6] = { 4, 5, 6, 1, 2, 3 };
    auto a = torch::from_blob(array, { 3, 2 }, torch::kInt);
    SPDLOG_DEBUG("a =\r\n{}", a);
    SPDLOG_DEBUG("a =\r\n{}", a.size(0));
    SPDLOG_DEBUG("a =\r\n{}", a.sizes());
}

static void testArgmax() {
    int array[6] = { 4, 5, 6, 1, 2, 3 };
    auto a = torch::from_blob(array, { 3, 2 }, torch::kInt);
    SPDLOG_DEBUG("a =\r\n{}", a);
    SPDLOG_DEBUG("a =\r\n{}", a.argmax(1));
    SPDLOG_DEBUG("a =\r\n{}", a.argmax(1, true));
}

static void testSqueeze() {
    int array[6] = { 4, 5, 6, 1, 2, 3 };
    auto a = torch::from_blob(array, { 3, 2 }, torch::kInt);
    SPDLOG_DEBUG("a =\r\n{}", a);
    // a = a.unsqueeze(0);
    a = a.unsqueeze(1);
    a = a.unsqueeze(2);
    SPDLOG_DEBUG("a =\r\n{}", a);
    SPDLOG_DEBUG("a =\r\n{}", a.squeeze());
}

static void testPermute() {
    torch::Tensor a = torch::linspace(1, 30, 30).view({ 3, 2, 5 });
    SPDLOG_DEBUG("a =\r\n{}", a.sizes());
    SPDLOG_DEBUG("a =\r\n{}", a.permute({ 0, 1, 2 }).sizes());
    SPDLOG_DEBUG("a =\r\n{}", a.permute({ 0, 2, 1 }).sizes());
    SPDLOG_DEBUG("a =\r\n{}", a.permute({ 1, 0, 2 }).sizes());
    SPDLOG_DEBUG("a =\r\n{}", a.permute({ 1, 2, 0 }).sizes());
    SPDLOG_DEBUG("a =\r\n{}", a.permute({ 2, 0, 1 }).sizes());
    SPDLOG_DEBUG("a =\r\n{}", a.permute({ 2, 1, 0 }).sizes());
    SPDLOG_DEBUG("a =\r\n{}", a);
    SPDLOG_DEBUG("a =\r\n{}", a.permute({ 0, 1, 2 }));
    SPDLOG_DEBUG("a =\r\n{}", a.permute({ 0, 2, 1 }));
    SPDLOG_DEBUG("a =\r\n{}", a.permute({ 1, 0, 2 }));
    SPDLOG_DEBUG("a =\r\n{}", a.permute({ 1, 2, 0 }));
    SPDLOG_DEBUG("a =\r\n{}", a.permute({ 2, 0, 1 }));
    SPDLOG_DEBUG("a =\r\n{}", a.permute({ 2, 1, 0 }));
}

static void testImage() {
    cv::Mat image = cv::imread("D://tmp/logo.png");
    cv::resize(image, image, cv::Size(200, 200));
    torch::Tensor a = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte);
    SPDLOG_DEBUG("a = \r\n{}", a.sizes());
    SPDLOG_DEBUG("a = \r\n{}", a.permute({2, 0, 1}).sizes());
    SPDLOG_DEBUG("a = \r\n{}", a.sizes());
    SPDLOG_DEBUG("a = \r\n{}", a.dim());
    // torch::Tensor b = torch::max_pool2d(a, 2);
    // cv::imwrite("D://tmp/logo.max.png", b);
}

static void testFind() {
    int array[12] = {
        4,  5,  6,
        14, 35, 16,
        24, 25, 26,
        1,  2,  3
    };
    auto a = torch::from_blob(array, { 4, 3 }, torch::kInt);
    SPDLOG_DEBUG("a = \r\n{}", a);
    SPDLOG_DEBUG("a sizes  = \r\n{}", a.sizes());
    SPDLOG_DEBUG("a slice  = \r\n{}", a.slice(0, 0, 2));
    SPDLOG_DEBUG("a slice  = \r\n{}", a.slice(1, 0, 2));
    auto [ key, value ] = a.max(1);
    SPDLOG_DEBUG("a max    = \r\n{}", key);
    SPDLOG_DEBUG("a max    = \r\n{}", value);
    SPDLOG_DEBUG("a argmax = \r\n{}", a.argmax(1));
    SPDLOG_DEBUG("a unsqueeze = \r\n{}", a.unsqueeze(0));
    SPDLOG_DEBUG("a unsqueeze = \r\n{}", a.unsqueeze(1));
}

static void testTransfor() {
    int array[12] = {
        4,  5,  6,
        14, 35, 16,
        24, 25, 26,
        1,  2,  3
    };
    auto a = torch::from_blob(array, { 4, 3 }, torch::kInt);
    SPDLOG_DEBUG("a = \r\n{}", a);
    SPDLOG_DEBUG("a = \r\n{}", a[0, 1]);
    // SPDLOG_DEBUG("a = \r\n{}", a[0, 2]);
    SPDLOG_DEBUG("a = \r\n{}", a[1, 1]);
    // SPDLOG_DEBUG("a = \r\n{}", a[1, 2]);
    SPDLOG_DEBUG("a = \r\n{}", a[2, 1]);
    // SPDLOG_DEBUG("a = \r\n{}", a[2, 2]);
    SPDLOG_DEBUG("a = \r\n{}", a.slice(0, 1));
    SPDLOG_DEBUG("a = \r\n{}", a.slice(1, 1));
    SPDLOG_DEBUG("a = \r\n{}", a.view(12));
    SPDLOG_DEBUG("a = \r\n{}", a.view({4, 3, 1, 1}));
    SPDLOG_DEBUG("a = \r\n{}", a[0][1].template item<int>());
}

static void testEqual() {
    torch::Tensor a = torch::rand({ 2, 3 });
    torch::Tensor b = a.view({ 2, 3 });
    SPDLOG_DEBUG("a = {}", a);
    SPDLOG_DEBUG("a = {}", b);
    SPDLOG_DEBUG("a = {}", (long long) &a);
    SPDLOG_DEBUG("a = {}", (long long) &b);
    SPDLOG_DEBUG("a = b: {}", a == b);
    int array[6] = { 4, 5, 6, 1, 2, 3 };
    auto ax = torch::from_blob(array, { 3, 2 }, torch::kInt);
    ax += 100;
    SPDLOG_DEBUG("ax = {}", ax);
    for(auto& v : array) {
        SPDLOG_DEBUG("axv = {}", v);
    }
}

static void testCUDA() {
    if(!torch::cuda::is_available()) {
        return;
    }
    auto device = torch::kCUDA;
    torch::tensor({33, 22, 11}).to(device);
}

static void testGrad() {
    torch::Tensor a = torch::ones({2, 3});
    // a.detach();
    // a.requires_grad = true;
    // a.set_requires_grad(false);
    a.set_requires_grad(true);
    try {
        // a.backward();
        auto b = torch::rand({2, 3});
        SPDLOG_DEBUG("a = {}", a);
        SPDLOG_DEBUG("b = {}", b);
        a.backward(b);
        // a.backward(torch::zeros({2, 3}));
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    // {
    //     torch::NoGradGuard noGradGuard;
    //     // 不算梯度
    // }
    SPDLOG_DEBUG("a grad = {}", a.grad());
    // a.is_leaf();
    // SPDLOG_DEBUG("a grad = {}", a.grad_fn());
    try {
        auto xa = torch::tensor({1, 2, 3, 4}, torch::kFloat64);
        xa.set_requires_grad(true);
        auto xb = 2 * xa;
        auto xc = xb.view({2, 2});
        SPDLOG_DEBUG("xc = {}", xc);
        auto xd = torch::tensor({{1.0, 0.1}, {0.01, 0.001}});
        SPDLOG_DEBUG("xd = {}", xd);
        xc.backward(xd);
        SPDLOG_DEBUG("xa grad = {}", xa.grad());
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
}

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    lifuren::testTensor();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
