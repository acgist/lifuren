#include "../src/header/MLPack.hpp"

int main(int argc, char const *argv[]) {
    lifuren::init(argc, argv);
    LOG(INFO) << "测试";
    lifuren::testMLPackLoadFile("D:\\tmp\\ml\\iris-data.csv");
    LOG(INFO) << "完成";
    lifuren::shutdown();
    return 0;
}
