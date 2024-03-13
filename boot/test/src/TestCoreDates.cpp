#include "utils/Dates.hpp"

void cost();

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    const std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    SPDLOG_DEBUG("当前时间：{}", now);
    const std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&timestamp);
    const std::string datetime = lifuren::dates::format(now, LFR_DATE_TIME_FORMAT);
    SPDLOG_DEBUG("日期时间转为字符串：{}", datetime);
    SPDLOG_DEBUG("日期时间转为字符串：{}", lifuren::dates::format(tm, LFR_DATE_TIME_FORMAT));
    tm = lifuren::dates::parseTm(datetime, LFR_DATE_TIME_FORMAT);
    SPDLOG_DEBUG("字符串转为日期时间：{}", lifuren::dates::parseTp(datetime, LFR_DATE_TIME_FORMAT));
    SPDLOG_DEBUG("字符串转为日期时间：{} - {} - {} - {} - {} - {}", 1900 + tm.tm_year, 1 + tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    const uint64_t millis = lifuren::dates::toMillis(now);
    SPDLOG_DEBUG("日期时间转时间戳：{}", millis);
    SPDLOG_DEBUG("日期时间转时间戳：{}", lifuren::dates::toMillis(tm));
    tm = lifuren::dates::toDatetimeTm(millis);
    SPDLOG_DEBUG("时间戳转日期时间：{}", lifuren::dates::toDatetimeTp(millis));
    SPDLOG_DEBUG("时间戳转日期时间：{} - {} - {} - {} - {} - {}", 1900 + tm.tm_year, 1 + tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    cost();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}

void cost() {
    const std::chrono::system_clock::time_point a = std::chrono::system_clock::now();
    const uint64_t millis = lifuren::dates::toMillis(a);
    for(int index = 0; index < 100000; ++index) {
        // lifuren::dates::toMillis(a);
        // lifuren::dates::toDatetimeTm(millis);
        // lifuren::dates::toDatetimeTp(millis);
        lifuren::dates::format(a, LFR_DATE_TIME_FORMAT);
        // lifuren::dates::parseTm("2012-12-12 12:12:12", LFR_DATE_TIME_FORMAT);
        // lifuren::dates::parseTp("2012-12-12 12:12:12", LFR_DATE_TIME_FORMAT);
    }
    const std::chrono::system_clock::time_point z = std::chrono::system_clock::now();
    SPDLOG_DEBUG("耗时：{}", std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count());
}
