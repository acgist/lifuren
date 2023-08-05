#include "../header/LifurenGG.hpp"

void testGLog(int argc, char const *argv[]) {
    LOG(INFO)    << "info";
    LOG(WARNING) << "warning";
    LOG(ERROR)   << "error";
    // 输出堆栈：输出以后终止程序
    // LOG(FATAL)   << "fatal";
    VLOG(0)      << "info";
    VLOG(1)      << "warning";
    google::ShutdownGoogleLogging();
}

void testJson() {
    nlohmann::json json = nlohmann::json::parse(R"(
    {
        "pi"   : 3.141,
        "happy": true
    }
    )");
    LOG(INFO) << json["pi"];
    std::string strings[] = { "1", "2" };
    LOG(INFO) << lifuren::gg::toJSON(strings, 2);
    int ints[] = { 1, 2, 3 };
    LOG(INFO) << lifuren::gg::toJSON(ints);
    // 不能自动推到
    // LOG(INFO) << lifuren::gg::toJSON({ 1, 2, 3 }, 3);
}

void testString() {
    std::string format = "li{}ren{}";
    std::string args[] = { "fu", "!!" };
    lifuren::gg::format(format, args, 2);
    LOG(INFO) << format;
}
