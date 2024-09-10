#include "Test.hpp"
#include "lifuren/Client.hpp"

[[maybe_unused]] static void testCommand() {
    // lifuren::CommandClient client{ "ls" };
    lifuren::CommandClient client{ "netstat -ano" };
    // lifuren::CommandClient client{ "D:/tmp/sdc/sd.exe -m D:/tmp/sdc/v1-5-pruned-emaonly.ckpt -p 'flower' --steps 1 -o D:/tmp/sdc/output.jpg" };
    const int& code = client.execute();
    SPDLOG_DEBUG("ls = {} - {} - {}", code, client.getCode(), client.getResult());
}

LFR_TEST(
    testCommand();
);
