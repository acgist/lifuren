#include "../../header/LibTorch.hpp"

// 测试初始化
void testInit();
// 测试拷贝
void testClone();
// 测试变形
void testResize();
// 测试切片
void testSlice();
// 测试运算
void testOperator();

void lifuren::testLibTorchTensor() {
    SPDLOG_DEBUG("是否支持CUDA：{}", torch::cuda::is_available());
    SPDLOG_DEBUG("是否支持CUDNN：{}", torch::cuda::cudnn_is_available());
    // testInit();
    // testClone();
    testResize();
}

void testInit() {
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

void testClone() {
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

void testResize() {
    // TODO：flatten、view、reshape、permute、transpose、squeeze、unsqueeze
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

void testSlice() {
    // TODO：narrow、select、index、index_put_、index_select、slice
}

void testOperator() {
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
    // TODO：cat、stack
}
