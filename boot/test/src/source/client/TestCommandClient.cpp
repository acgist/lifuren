#include "lifuren/Client.hpp"

#include "lifuren/Logger.hpp"

#include "spdlog/spdlog.h"

[[maybe_unused]] static void testCommand() {
    // lifuren::CommandClient client{ "ls" };
    lifuren::CommandClient client{ "netstat -ano" };
    // lifuren::CommandClient client{ "D:/tmp/sdc/sd.exe -m D:/tmp/sdc/v1-5-pruned-emaonly.ckpt -p 'flower' --steps 1 -o D:/tmp/sdc/output.jpg" };
    const int& code = client.execute();
    SPDLOG_DEBUG("ls = {} - {} - {}", code, client.getCode(), client.getResult());
}

int main() {
    lifuren::logger::init();
    SPDLOG_DEBUG("测试");
    testCommand();
    SPDLOG_DEBUG("完成");
    lifuren::logger::shutdown();
    return 0;
}
