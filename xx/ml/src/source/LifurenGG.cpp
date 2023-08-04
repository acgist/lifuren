#include "../header/LifurenGG.hpp"

void testGLog(int argc, char const *argv[]) {
    std::filesystem::create_directories("logs");
    FLAGS_v                = 1;    // 输出级别
    FLAGS_alsologtostderr  = true; // 文件和控制台
    FLAGS_colorlogtostderr = true; // 颜色输出
    google::SetLogDestination(google::GLOG_INFO,    "logs/info_");
    google::SetLogDestination(google::GLOG_WARNING, "logs/warning_");
    google::SetLogDestination(google::GLOG_ERROR,   "logs/error_");
    google::SetLogDestination(google::GLOG_FATAL,   "logs/fatal_");
    google::SetLogFilenameExtension(".log");
    google::InitGoogleLogging(argc > 0 ? argv[0] : nullptr);
    LOG(INFO)    << "info";
    LOG(WARNING) << "warning";
    LOG(ERROR)   << "error";
    // LOG(FATAL)   << "fatal"; // 输出堆栈：输出以后终止程序
    VLOG(0)      << "info";
    VLOG(1)      << "warning";
    google::FlushLogFiles(google::GLOG_INFO);
    google::FlushLogFiles(google::GLOG_WARNING);
    google::FlushLogFiles(google::GLOG_ERROR);
    google::FlushLogFiles(google::GLOG_FATAL);
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
    // LOG(INFO) << lifuren::gg::toJSON({ 1, 2, 3 }, 3);
}

void testString() {
    std::string format = "li{}ren{}";
    std::string args[] = { "fu", "!!" };
    lifuren::gg::format(format, args, 2);
    LOG(INFO) << format;
}
