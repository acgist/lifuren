#include "lifuren/Lifuren.hpp"

#include <mutex>
#include <chrono>

static std::mutex mutex;

size_t lifuren::uuid() noexcept {
    static int index = 0;
    const static int MIN_INDEX = 0;
    const static int MAX_INDEX = 9999;
    auto timePoint = std::chrono::system_clock::now();
    auto timestamp = std::chrono::system_clock::to_time_t(timePoint);
    auto localtime = std::localtime(&timestamp);
    int i = 0;
    {
        std::lock_guard<std::mutex> lock(mutex);
        i = index;
        if(++index > MAX_INDEX) {
            index = MIN_INDEX;
        }
    }
    size_t id = 100000000000000 * (localtime->tm_year + 1900) +
                1000000000000   * (localtime->tm_mon  +    1) +
                10000000000     *  localtime->tm_mday         +
                100000000       *  localtime->tm_hour         +
                1000000         *  localtime->tm_min          +
                10000           *  localtime->tm_sec          +
                i;
    return id;
}
