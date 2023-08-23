#include "../header/Template.hpp"

void lifuren::lifuren() {
    VLOG(0)   << "李夫人的模板：off";
    VLOG(1)   << "李夫人的模板：error";
    VLOG(2)   << "李夫人的模板：warn";
    VLOG(3)   << "李夫人的模板：info";
    VLOG(4)   << "李夫人的模板：debug";
    LOG(INFO)    << "李夫人的模板";
    LOG(WARNING) << "李夫人的模板";
    LOG(ERROR)   << "李夫人的模板";
    LOG(FATAL)   << "李夫人的模板";
}