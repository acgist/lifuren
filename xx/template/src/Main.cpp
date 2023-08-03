#include "header/Template.hpp"

int main(int argc, char const* argv[]) {
    lifuren::glog::init(argc, argv);
    lifuren::lifuren();
    lifuren::glog::shutdown();
    return 0;
}
