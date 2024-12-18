#include "lifuren/Test.hpp"

#include "lifuren/Date.hpp"

#include "spdlog/fmt/chrono.h"

[[maybe_unused]] static void testFormat() {
    const std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    SPDLOG_DEBUG("当前时间：{}", now);
    SPDLOG_DEBUG("日期时间转为字符串：{}", lifuren::date::format(now, LFR_DATE_TIME_FORMAT));
    const std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&timestamp);
    SPDLOG_DEBUG("日期时间转为字符串：{}", lifuren::date::format(tm, LFR_DATE_TIME_FORMAT));
}

[[maybe_unused]] static void testParse() {
    std::string datetime = "2024-05-27 08:08:18";
    SPDLOG_DEBUG("字符串转为日期时间：{}", lifuren::date::parseTp(datetime, LFR_DATE_TIME_FORMAT));
    std::tm tm = lifuren::date::parseTm(datetime, LFR_DATE_TIME_FORMAT);
    SPDLOG_DEBUG("字符串转为日期时间：{} - {} - {} - {} - {} - {}", 1900 + tm.tm_year, 1 + tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

}

[[maybe_unused]] static void testMillis() {
    const std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    const uint64_t millis = lifuren::date::toMillis(now);
    std::tm tm = lifuren::date::parseTm(millis);
    SPDLOG_DEBUG("日期时间转时间戳：{}", lifuren::date::toMillis(now));
    SPDLOG_DEBUG("日期时间转时间戳：{}", lifuren::date::toMillis(tm));
    SPDLOG_DEBUG("时间戳转日期时间：{}", lifuren::date::parseTp(millis));
    SPDLOG_DEBUG("时间戳转日期时间：{} - {} - {} - {} - {} - {}", 1900 + tm.tm_year, 1 + tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
}

[[maybe_unused]] static void testCost() {
    const std::chrono::system_clock::time_point a = std::chrono::system_clock::now();
    // const uint64_t millis = lifuren::date::toMillis(a);
    for(int index = 0; index < 100000; ++index) {
        // lifuren::date::toMillis(a);
        // lifuren::date::parseTm(millis);
        // lifuren::date::parseTp(millis);
        // 优化50毫秒以内
        lifuren::date::format(a, LFR_DATE_TIME_FORMAT);
        // lifuren::date::parseTm("2012-12-12 12:12:12", LFR_DATE_TIME_FORMAT);
        // 优化100毫秒以内
        // lifuren::date::parseTp("2012-12-12 12:12:12", LFR_DATE_TIME_FORMAT);
    }
    const std::chrono::system_clock::time_point z = std::chrono::system_clock::now();
    SPDLOG_DEBUG("耗时：{}", std::chrono::duration_cast<std::chrono::milliseconds>((z - a)).count());
}

LFR_TEST(
    testFormat();
    testParse();
    testMillis();
    testCost();
);
