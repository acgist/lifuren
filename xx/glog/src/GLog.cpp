#include <iostream>
#include "glog/logging.h"

int main(int argc, char const *argv[]) {
    FLAGS_alsologtostderr  = true; // 文件和控制台
    FLAGS_colorlogtostderr = true; // 颜色输出
    google::SetLogDestination(google::GLOG_INFO,    "logs/info_");
    google::SetLogDestination(google::GLOG_WARNING, "logs/warning_");
    google::SetLogDestination(google::GLOG_ERROR,   "logs/error_");
    google::SetLogDestination(google::GLOG_FATAL,   "logs/fatal_");
    google::SetLogFilenameExtension(".log");
    google::InitGoogleLogging("lifuren");
    LOG(INFO)    << "info";
    LOG(WARNING) << "warning";
    LOG(ERROR)   << "error";
    LOG(FATAL)   << "fatal";
    google::FlushLogFiles(google::GLOG_INFO);
    google::FlushLogFiles(google::GLOG_WARNING);
    google::FlushLogFiles(google::GLOG_ERROR);
    google::FlushLogFiles(google::GLOG_FATAL);
    google::ShutdownGoogleLogging();
    return 0;
}
