#include "Test.hpp"

#include "lifuren/Strings.hpp"

static void testLowerUpper() {
    std::string value = "LIfuREN";
    lifuren::strings::toLower(value);
    assert("lifuren" == value);
    lifuren::strings::toUpper(value);
    assert("LIFUREN" == value);
    value = " 1234 ";
}

static void testStringTrim() {
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

static void testConstCharTrim() {
    char value1[]{' ', '1', '2', '3', '4', ' ', '\0'};
    char* trim  = lifuren::strings::trim(value1);
    assert(std::strcmp("1234", trim) == 0);
    char value2[]{' ', '1', '2', '3', '4', '\0'};
    trim = lifuren::strings::trim(value2);
    assert(std::strcmp("1234", trim) == 0);
    char value3[]{'1', '2', '3', '4', ' ', '\0'};
    trim = lifuren::strings::trim(value3);
    assert(std::strcmp("1234", trim) == 0);
    char value4[]{'1', '2', '3', '4', '\0'};
    trim = lifuren::strings::trim(value4);
    assert(std::strcmp("1234", trim) == 0);
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

static void testToChars() {
    auto&& vector = lifuren::strings::toChars("你好啊！abc测试1234 1234李夫人?!好吧");
    for(const auto& v : vector) {
        SPDLOG_DEBUG("char = {}", v);
    }
}

static void testJoin() {
    std::vector<std::string> vector;
    SPDLOG_DEBUG("join：{}", lifuren::strings::join(vector, ","));
    vector.push_back("1");
    vector.push_back("2");
    vector.push_back("3");
    SPDLOG_DEBUG("join：{}", lifuren::strings::join(vector, ","));
}

static void testSplit() {
    std::vector<std::string> split = lifuren::strings::split("", ",", false, false);
    assert(split.size() == 1);
    split = lifuren::strings::split("1", ",");
    assert(split.size() == 1);
    split = lifuren::strings::split(",,", ",", false, false);
    assert(split.size() == 3);
    split = lifuren::strings::split(",,", ",");
    assert(split.size() == 0);
    split = lifuren::strings::split("1,", ",");
    assert(split.size() == 1);
    split = lifuren::strings::split(",1", ",", false, false);
    assert(split.size() == 2);
    split = lifuren::strings::split(",1,", ",", false, false);
    assert(split.size() == 3);
    split = lifuren::strings::split(",1,", ",");
    assert(split.size() == 1);
    split = lifuren::strings::split("1,2", ",");
    assert(split.size() == 2);
    split = lifuren::strings::split(",1,2", ",", false, false);
    assert(split.size() == 3);
    split = lifuren::strings::split("1,2,", ",", false, false);
    assert(split.size() == 3);
    split = lifuren::strings::split(",1,2,", ",", false, false);
    assert(split.size() == 4);
    split = lifuren::strings::split(",1,2,", ",");
    assert(split.size() == 2);
    split = lifuren::strings::split("1，2", "，");
    assert(split.size() == 2);
}

static void testSplitMulti() {
    std::vector<std::string> split = lifuren::strings::split("", std::vector<std::string>{ "，", "。" }, false, false);
    assert(split.size() == 1);
    split = lifuren::strings::split("1", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 1);
    split = lifuren::strings::split("，1", std::vector<std::string>{ "，", "。" }, false, false);
    assert(split.size() == 2);
    split = lifuren::strings::split("1，", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 1);
    split = lifuren::strings::split("，1，", std::vector<std::string>{ "，", "。" }, false, false);
    assert(split.size() == 3);
    split = lifuren::strings::split("，1，", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 1);
    split = lifuren::strings::split("1，2。3", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 3);
    split = lifuren::strings::split("。1，2。3。", std::vector<std::string>{ "，", "。" }, false, false);
    assert(split.size() == 5);
    split = lifuren::strings::split("。1，2。3。", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 3);
}

LFR_TEST(
    // testLowerUpper();
    // testStringTrim();
    // testConstCharTrim();
    // testLength();
    // testSubstr();
    // testReplace();
    testToChars();
    testJoin();
    testSplit();
    testSplitMulti();
);
