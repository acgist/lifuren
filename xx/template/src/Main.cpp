#include "header/Template.hpp"

int main(int argc, char const* argv[]) {
    lifuren::gg::init(argc, argv);
    lifuren::lifuren();
    lifuren::gg::shutdown();
    return 0;
}
