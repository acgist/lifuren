/**
 * 李夫人 - 玉簪花神
 * 
 * @author acgist
 */
#include "GLog.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::init(argc, argv);
    LOG(INFO) << "启动系统";
    LOG(INFO) << "启动完成";
    lifuren::shutdown();
    return 0;
}
