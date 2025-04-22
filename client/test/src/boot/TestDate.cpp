#include "lifuren/Test.hpp"

#include "lifuren/Date.hpp"

#include "spdlog/fmt/chrono.h"

[[maybe_unused]] static void testFormat() {
    const std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    const std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&timestamp);
    SPDLOG_DEBUG("当前时间：{}", now);
    SPDLOG_DEBUG("日期时间转为字符串：{}", lifuren::date::format(now, LFR_DATE_TIME_FORMAT));
    SPDLOG_DEBUG("日期时间转为字符串：{}", lifuren::date::format(tm,  LFR_DATE_TIME_FORMAT));
}

[[maybe_unused]] static void testParse() {
    std::string datetime = "2024-05-27 18:08:18";
    auto date  = lifuren::date::parse_time_point(datetime, LFR_DATE_TIME_FORMAT);
    std::tm tm = lifuren::date::parse_tm        (datetime, LFR_DATE_TIME_FORMAT);
    SPDLOG_DEBUG("字符串转为日期时间：{}", date);
    SPDLOG_DEBUG("字符串转为日期时间：{}-{:#02d}-{:#02d} {:#02d}:{:#02d}:{:#02d}", 1900 + tm.tm_year, 1 + tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
}

[[maybe_unused]] static void testMillis() {
    const std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    const uint64_t millis = lifuren::date::to_millis(now);
    std::tm tm = lifuren::date::parse_tm(millis);
    SPDLOG_DEBUG("日期时间转时间戳：{}", lifuren::date::to_millis(now));
    SPDLOG_DEBUG("日期时间转时间戳：{}", lifuren::date::to_millis(tm));
    SPDLOG_DEBUG("时间戳转日期时间：{}-{:#02d}-{:#02d} {:#02d}:{:#02d}:{:#02d}", 1900 + tm.tm_year, 1 + tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
}

[[maybe_unused]] static void testPerformance() {
    const std::string date = "2012-12-12 12:12:12";
    const std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    const uint64_t millis = lifuren::date::to_millis(now);
    const std::tm tm = lifuren::date::parse_tm(millis);
    lifuren::test::loop(100'000, [&tm, &now, date, millis]() {
        lifuren::date::format(tm, LFR_DATE_TIME_FORMAT);
        // lifuren::date::format(now, LFR_DATE_TIME_FORMAT);
        // lifuren::date::to_millis(now);
        // lifuren::date::parse_tm(millis);
        // lifuren::date::parse_tm(date, LFR_DATE_TIME_FORMAT);
        // lifuren::date::parse_time_point(millis);
        // lifuren::date::parse_time_point(date, LFR_DATE_TIME_FORMAT);
    });
}

LFR_TEST(
    testFormat();
    testParse();
    testMillis();
    testPerformance();
);
