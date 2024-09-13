#include "Test.hpp"

#include "lifuren/EmbeddingClient.hpp"

[[maybe_unused]] static void testEmbedding() {
    lifuren::ChineseWordVectorsEmbeddingClient client{};
    auto v = client.getVector("中");
    SPDLOG_DEBUG("v = {}", v.size());
}

[[maybe_unused]] static void testSegmentEmbedding() {
    lifuren::ChineseWordVectorsEmbeddingClient client{};
    auto v = client.getSegmentVector(R"(
天道不可问，问天天杳冥。
如何正月霜，百卉皆凋零。
爝火乱白日，夜光杂飞萤。
贾生贤大夫，所以离汉庭。
河色本异洛，渭流颇殊泾。
清浊共朝宗，滔滔曾莫停。
巨川用舟楫，傅说匡武丁。
大船胶潢洿，不使济沧溟。
中夜击唾壶，仰头望天庚。
三台位尚阙，群象徒荧荧。
鸾翮时暂铩，龙门昼常扃。
心倾下士尽，眼顾贫交青。
幽室养虚白，香茶陶性灵。
应将混荣辱，讵肯甘膻腥。
至道是吾本，浮云劳我形。
手中菩提子，身外《莲花经》。
兴来步出门，长啸临江亭。
毫端洒垂露，赋里摇文星。
大音比叫钟，大智同挈缾。
巴歌利节曲，布鼓随雷霆。
陋巷一簟食，在原双鹡[鸰]（领）。
每怀受施恩，长记座右铭。
食惠饱复饭，饮仁醉还醒。
绵绵寄生叶，泛泛无根萍。
萍岂不随流，爱君水清泠。
叶岂不恋本。
爱君树芳馨。
彷徨窃三省，感激终百龄。
开合眷已重，扫门心匪宁。
铅刀冀效割，钝刃思发铏。
危弦托在兹，实愿知音听。
（影印本《诗渊》第一册第四七九至四八○页）。
    )");
    SPDLOG_DEBUG("v = {}", v.size());
}

[[maybe_unused]] static void testRelease() {
    lifuren::ChineseWordVectorsEmbeddingClient client{};
    client.release();
}

LFR_TEST(
    testEmbedding();
    testSegmentEmbedding();
    testRelease();
);
