#include "Test.hpp"

#include "lifuren/Poetrys.hpp"

[[maybe_unused]] static void testPoetrys() {
    auto ret = lifuren::poetrys::beautify(R"(
    春花秋月何时了，往事知多少？小楼昨夜又东风，故国不堪回首月明中。
    雕栏玉砌应犹在，只是朱颜改。问君能有几多愁？恰似一江春水向东流。
    )");
    SPDLOG_DEBUG("美化：\n{}", ret);
}

LFR_TEST(
    testPoetrys();
);
