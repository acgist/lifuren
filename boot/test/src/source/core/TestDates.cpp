#include "lifuren/Dates.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"
#include "spdlog/fmt/chrono.h"

static void testFormat() {
    const std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    SPDLOG_DEBUG("当前时间：{}", now);
    SPDLOG_DEBUG("日期时间转为字符串：{}", lifuren::dates::format(now, LFR_DATE_TIME_FORMAT));
    const std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&timestamp);
    SPDLOG_DEBUG("日期时间转为字符串：{}", lifuren::dates::format(tm, LFR_DATE_TIME_FORMAT));
}

static void testParse() {
    std::string datetime = "2024-05-27 08:08:18";
    SPDLOG_DEBUG("字符串转为日期时间：{}", lifuren::dates::parseTp(datetime, LFR_DATE_TIME_FORMAT));
    std::tm tm = lifuren::dates::parseTm(datetime, LFR_DATE_TIME_FORMAT);
    SPDLOG_DEBUG("字符串转为日期时间：{} - {} - {} - {} - {} - {}", 1900 + tm.tm_year, 1 + tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

}

static void testMillis() {
    const std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    const uint64_t millis = lifuren::dates::toMillis(now);
    std::tm tm = lifuren::dates::parseTm(millis);
    SPDLOG_DEBUG("日期时间转时间戳：{}", lifuren::dates::toMillis(now));
    SPDLOG_DEBUG("日期时间转时间戳：{}", lifuren::dates::toMillis(tm));
    SPDLOG_DEBUG("时间戳转日期时间：{}", lifuren::dates::parseTp(millis));
    SPDLOG_DEBUG("时间戳转日期时间：{} - {} - {} - {} - {} - {}", 1900 + tm.tm_year, 1 + tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
}

static void testCost() {
    const std::chrono::system_clock::time_point a = std::chrono::system_clock::now();
    const uint64_t millis = lifuren::dates::toMillis(a);
    for(int index = 0; index < 100000; ++index) {
        // lifuren::dates::toMillis(a);
        // lifuren::dates::parseTm(millis);
        // lifuren::dates::parseTp(millis);
        // 优化50毫秒以内
        lifuren::dates::format(a, LFR_DATE_TIME_FORMAT);
        // lifuren::dates::parseTm("2012-12-12 12:12:12", LFR_DATE_TIME_FORMAT);
        // 优化100毫秒以内
        // lifuren::dates::parseTp("2012-12-12 12:12:12", LFR_DATE_TIME_FORMAT);
    }
    const std::chrono::system_clock::time_point z = std::chrono::system_clock::now();
    SPDLOG_DEBUG("耗时：{}", std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count());
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testFormat();
    testParse();
    testMillis();
    testCost();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
