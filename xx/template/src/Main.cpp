#include "header/GLog.hpp"
#include "header/Template.hpp"

int main(int argc, char const* argv[]) {
    lifuren::initGlog(argc, argv);
    lifuren::lifuren();
    lifuren::shutdownGlog();
    return 0;
}
