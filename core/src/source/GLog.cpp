#include "../header/GLog.hpp"

void lifuren::init(const int argc, const char * const argv[]) {
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
    LOG(INFO) << R"(
        
        北方有佳人，绝世而独立。
        一顾倾人城，再顾倾人国。
        宁不知倾城与倾国，佳人难再得。
    )";
}

void lifuren::shutdown() {
    google::FlushLogFiles(google::GLOG_INFO);
    google::FlushLogFiles(google::GLOG_WARNING);
    google::FlushLogFiles(google::GLOG_ERROR);
    google::FlushLogFiles(google::GLOG_FATAL);
    google::ShutdownGoogleLogging();
}
