#include "lifuren/Test.hpp"

#include <cassert>

#include "lifuren/String.hpp"

[[maybe_unused]] static void testJoin() {
    std::vector<std::string> vector;
    assert("" == lifuren::string::join(vector, ","));
    vector.push_back("1");
    vector.push_back("2");
    vector.push_back("3");
    assert("1,2,3" == lifuren::string::join(vector, ","));
}

[[maybe_unused]] static void testSplit() {
    std::vector<std::string> split = lifuren::string::split("", ",", false, false);
    assert(split.size() == 1);
    split = lifuren::string::split("1", ",");
    assert(split.size() == 1);
    split = lifuren::string::split(",,", ",", false, false);
    assert(split.size() == 3);
    split = lifuren::string::split(",,", ",");
    assert(split.size() == 0);
    split = lifuren::string::split("1,", ",");
    assert(split.size() == 1);
    split = lifuren::string::split(",1", ",", false, false);
    assert(split.size() == 2);
    split = lifuren::string::split(",1,", ",", false, false);
    assert(split.size() == 3);
    split = lifuren::string::split(",1,", ",");
    assert(split.size() == 1);
    split = lifuren::string::split("1,2", ",");
    assert(split.size() == 2);
    split = lifuren::string::split(",1,2", ",", false, false);
    assert(split.size() == 3);
    split = lifuren::string::split("1,2,", ",", false, false);
    assert(split.size() == 3);
    split = lifuren::string::split(",1,2,", ",", false, false);
    assert(split.size() == 4);
    split = lifuren::string::split(",1,2,", ",");
    assert(split.size() == 2);
    split = lifuren::string::split("1，2", "，");
    assert(split.size() == 2);
}

[[maybe_unused]] static void testSplitMulti() {
    std::vector<std::string> split = lifuren::string::split("", std::vector<std::string>{ "，", "。" }, false, false);
    assert(split.size() == 1);
    split = lifuren::string::split("1", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 1);
    split = lifuren::string::split("，1", std::vector<std::string>{ "，", "。" }, false, false);
    assert(split.size() == 2);
    split = lifuren::string::split("1，", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 1);
    split = lifuren::string::split("，1，", std::vector<std::string>{ "，", "。" }, false, false);
    assert(split.size() == 3);
    split = lifuren::string::split("，1，", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 1);
    split = lifuren::string::split("1，2。3", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 3);
    split = lifuren::string::split("。1，2。3。", std::vector<std::string>{ "，", "。" }, false, false);
    assert(split.size() == 5);
    split = lifuren::string::split("。1，2。3。", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 3);
}

[[maybe_unused]] static void testLowerUpper() {
    std::string value = "LIfuREN";
    lifuren::string::toLower(value);
    assert("lifuren" == value);
    lifuren::string::toUpper(value);
    assert("LIFUREN" == value);
    value = " 1234 ";
}

[[maybe_unused]] static void testTrim() {
    // string
    std::string value = " 1234 ";
    std::string trim  = lifuren::string::trim(value);
    assert("1234" == trim);
    value = " 1234";
    trim = lifuren::string::trim(value);
    assert("1234" == trim);
    value = "1234 ";
    trim = lifuren::string::trim(value);
    assert("1234" == trim);
    value = "1234";
    trim = lifuren::string::trim(value);
    assert("1234" == trim);
    // char*
    char value1[]{' ', '1', '2', '3', '4', ' ', '\0'};
    char* chars  = lifuren::string::trim(value1);
    assert(std::strcmp("1234", chars) == 0);
    char value2[]{' ', '1', '2', '3', '4', '\0'};
    chars = lifuren::string::trim(value2);
    assert(std::strcmp("1234", chars) == 0);
    char value3[]{'1', '2', '3', '4', ' ', '\0'};
    chars = lifuren::string::trim(value3);
    assert(std::strcmp("1234", chars) == 0);
    char value4[]{'1', '2', '3', '4', '\0'};
    chars = lifuren::string::trim(value4);
    assert(std::strcmp("1234", chars) == 0);
}

[[maybe_unused]] static void testLength() {
    assert(2 == lifuren::string::length("12"));
    assert(2 == lifuren::string::length("1f"));
    assert(4 == lifuren::string::length("测hi试"));
}

[[maybe_unused]] static void testSubstr() {
    assert("1234" == lifuren::string::substr("12345",  0, 4));
    assert("1234" == lifuren::string::substr("d12345", 1, 4));
    assert(std::string("") == lifuren::string::substr("这是一个测试", 0, 0));
    assert(std::string("这") == lifuren::string::substr("这是一个测试", 0, 1));
    assert(std::string("是") == lifuren::string::substr("这是一个测试", 1, 1));
    assert(std::string("是一") == lifuren::string::substr("这是一个测试", 1, 2));
    assert(std::string("是1个测") == lifuren::string::substr("这是1个测试", 1, 4));
    assert(std::string("是一个测") == lifuren::string::substr("这是一个测试", 1, 4));
    assert(std::string("是一个测试") == lifuren::string::substr("这是一个测试", 1, 40));
}

[[maybe_unused]] static void testToChars() {
    auto&& vector = lifuren::string::toChars("你好啊！abc测试1234 1234李夫人?!好吧");
    std::vector<std::string> diff{ "你", "好", "啊", "！", "a", "b", "c", "测", "试", "1", "2", "3", "4", "1", "2", "3", "4", "李", "夫", "人", "?", "!", "好", "吧" };
    assert(diff == vector);
}

[[maybe_unused]] static void testReplace() {
    std::string value = "12测1试35";
    lifuren::string::replace(value, "1试", "4");
    assert("12测435" == value);
    value = "12测1试35";
    std::vector<std::string> multi = { "5", "1试" };
    lifuren::string::replace(value, multi, "4");
    assert("12测434" == value);
}

LFR_TEST(
    testJoin();
    testSplit();
    testSplitMulti();
    testLowerUpper();
    testTrim();
    testLength();
    testSubstr();
    testToChars();
    testReplace();
);
