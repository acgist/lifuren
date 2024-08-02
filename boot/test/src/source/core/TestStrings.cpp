#include "lifuren/Strings.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

static void testLowerUpper() {
    std::string value = "LIfuREN";
    lifuren::strings::toLower(value);
    assert("lifuren" == value);
    lifuren::strings::toUpper(value);
    assert("LIFUREN" == value);
    value = " 1234 ";
}

static void testTrim() {
    std::string value = " 1234 ";
    std::string trim  = lifuren::strings::trim(value);
    assert("1234" == trim);
    value = " 1234";
    trim = lifuren::strings::trim(value);
    assert("1234" == trim);
    value = "1234 ";
    trim = lifuren::strings::trim(value);
    assert("1234" == trim);
    value = "1234";
    trim = lifuren::strings::trim(value);
    assert("1234" == trim);
}

static void testLength() {
    assert(2 == lifuren::strings::length("12"));
    assert(2 == lifuren::strings::length("1f"));
    assert(4 == lifuren::strings::length("测hi试"));
}

static void testSubstr() {
    assert("1234" == lifuren::strings::substr("12345",  0, 4));
    assert("1234" == lifuren::strings::substr("d12345", 1, 4));
}

static void testReplace() {
    std::string value = "1235";
    lifuren::strings::replace(value, "5", "4");
    assert("1234" == value);
    value = "1235";
    std::vector<std::string> multi = { "5" };
    lifuren::strings::replace(value, multi, "4");
    assert("1234" == value);
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testLowerUpper();
    testTrim();
    testLength();
    testSubstr();
    testReplace();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}