#include "Test.hpp"
#include "lifuren/Lifuren.hpp"

LFR_TEST(
    // yyyyMMddHHmmssxxxx
    SPDLOG_DEBUG("uuid = {}", lifuren::uuid());
    SPDLOG_DEBUG("uuid = {}", lifuren::uuid());
    SPDLOG_DEBUG("uuid = {}", lifuren::uuid());
    SPDLOG_DEBUG("uuid = {}", lifuren::uuid());
);
