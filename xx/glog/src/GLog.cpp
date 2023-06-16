#include <filesystem>

#include "glog/logging.h"

int main(int argc, char const *argv[]) {
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
    // 输出堆栈
    // LOG(FATAL)   << "fatal";
    VLOG(0)      << "info";
    VLOG(1)      << "warning";
    google::FlushLogFiles(google::GLOG_INFO);
    google::FlushLogFiles(google::GLOG_WARNING);
    google::FlushLogFiles(google::GLOG_ERROR);
    google::FlushLogFiles(google::GLOG_FATAL);
    google::ShutdownGoogleLogging();
    return 0;
}
