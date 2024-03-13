#include <regex>
#include <string>
#include <vector>

#include "Logger.hpp"
#include "utils/Collections.hpp"

int main(const int argc, const char * const argv[]) {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    std::vector<std::string> vector;
    SPDLOG_DEBUG("join：{}", lifuren::collections::join(vector, ","));
    vector.push_back("1");
    vector.push_back("2");
    vector.push_back("3");
    SPDLOG_DEBUG("join：{}", lifuren::collections::join(vector, ","));
    std::vector<std::string> split = lifuren::collections::split("", ",");
    assert(split.size() == 1);
    split = lifuren::collections::split("1", ",");
    assert(split.size() == 1);
    split = lifuren::collections::split(",,", ",");
    assert(split.size() == 3);
    split = lifuren::collections::split("1,", ",");
    assert(split.size() == 2);
    split = lifuren::collections::split(",1", ",");
    assert(split.size() == 2);
    split = lifuren::collections::split(",1,", ",");
    assert(split.size() == 3);
    split = lifuren::collections::split("1,2", ",");
    assert(split.size() == 2);
    split = lifuren::collections::split(",1,2", ",");
    assert(split.size() == 3);
    split = lifuren::collections::split("1,2,", ",");
    assert(split.size() == 3);
    split = lifuren::collections::split(",1,2,", ",");
    assert(split.size() == 4);
    split = lifuren::collections::split("1，2", "，");
    assert(split.size() == 2);
    split = lifuren::collections::split("", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 1);
    split = lifuren::collections::split("1", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 1);
    split = lifuren::collections::split("，1", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 2);
    split = lifuren::collections::split("1，", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 2);
    split = lifuren::collections::split("，1，", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 3);
    split = lifuren::collections::split("1，2。3", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 3);
    split = lifuren::collections::split("。1，2。3。", std::vector<std::string>{ "，", "。" });
    assert(split.size() == 5);
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}


