/**
 * 机器学习测试代码
 * 
 * @author acgist
 */
#include "GLog.hpp"

int main(int argc, char const* argv[]) {
    lifuren::init(argc, argv);
    LOG(INFO) << "测试";
    LOG(INFO) << "完成";
    lifuren::shutdown();
    return 0;
}
