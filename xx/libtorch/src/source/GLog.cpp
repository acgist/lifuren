#include "../header/Glog.hpp"

namespace lifuren {

void initGlog(int argc, char const* argv[]) {
    std::filesystem::create_directories("logs");
    FLAGS_v                = 4;    // 输出级别
    FLAGS_alsologtostderr  = true; // 文件和控制台
    FLAGS_colorlogtostderr = true; // 颜色输出
    google::SetLogDestination(google::GLOG_INFO,    "logs/info_");
    google::SetLogDestination(google::GLOG_WARNING, "logs/warning_");
    google::SetLogDestination(google::GLOG_ERROR,   "logs/error_");
    google::SetLogDestination(google::GLOG_FATAL,   "logs/fatal_");
    google::SetLogFilenameExtension(".log");
    google::InitGoogleLogging(argc > 0 ? argv[0] : nullptr);
}

void shutdownGlog() {
    google::FlushLogFiles(google::GLOG_INFO);
    google::FlushLogFiles(google::GLOG_WARNING);
    google::FlushLogFiles(google::GLOG_ERROR);
    google::FlushLogFiles(google::GLOG_FATAL);
    google::ShutdownGoogleLogging();
}

}