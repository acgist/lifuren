#include "../header/GLog.hpp"

namespace lifuren {

namespace gg {

    void init(int argc, char const* argv[]) {
        std::filesystem::create_directories("logs");
        FLAGS_v                = 4;
        FLAGS_alsologtostderr  = true;
        FLAGS_colorlogtostderr = true;
        google::SetLogDestination(google::GLOG_INFO,    "logs/info_");
        google::SetLogDestination(google::GLOG_WARNING, "logs/warning_");
        google::SetLogDestination(google::GLOG_ERROR,   "logs/error_");
        google::SetLogDestination(google::GLOG_FATAL,   "logs/fatal_");
        google::SetLogFilenameExtension(".log");
        google::InitGoogleLogging(argc > 0 ? argv[0] : nullptr);
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    }

    void shutdown() {
        google::FlushLogFiles(google::GLOG_INFO);
        google::FlushLogFiles(google::GLOG_WARNING);
        google::FlushLogFiles(google::GLOG_ERROR);
        google::FlushLogFiles(google::GLOG_FATAL);
        google::ShutdownGoogleLogging();
    }

}

}