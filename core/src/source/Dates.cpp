#include "lifuren/Dates.hpp"

#include <ctime>
#include <iomanip>
#include <sstream>

#ifndef LFR_DATE_FORMAT_STREAM
#define LFR_DATE_FORMAT_STREAM false
#endif

std::string lifuren::dates::format(const std::tm& datetime, const std::string& format) {
    #if LFR_DATE_FORMAT_STREAM
    // 性能较差
    std::ostringstream output;
    output << std::put_time(&datetime, format.c_str());
    return output.str();
    #else
    // 性能较高
    char output[20];
    std::strftime(output, 20, format.c_str(), &datetime);
    return output;
    #endif
}

std::string lifuren::dates::format(const std::chrono::system_clock::time_point& datetime, const std::string& format) {
    const std::time_t timestamp = std::chrono::system_clock::to_time_t(datetime);
    const std::tm* tm = std::localtime(&timestamp);
    return lifuren::dates::format(*tm, format);
}

std::tm lifuren::dates::parseTm(const std::string& datetime, const std::string& format) {
    std::tm tm;
    #ifdef _WIN32
    std::istringstream input(datetime);
	input >> std::get_time(&tm, format.c_str());
    #else
    strptime(datetime.c_str(), format.c_str(), &tm);
    #endif
    return tm;
}

std::chrono::system_clock::time_point lifuren::dates::parseTp(const std::string& datetime, const std::string& format) {
	std::tm tm = lifuren::dates::parseTm(datetime, format);
	const std::time_t timestamp = std::mktime(&tm);
	return std::chrono::system_clock::from_time_t(timestamp);
}

uint64_t lifuren::dates::toMillis(std::tm& datetime) {
    const std::time_t timestamp = std::mktime(&datetime);
    return lifuren::dates::toMillis(std::chrono::system_clock::from_time_t(timestamp));
}

uint64_t lifuren::dates::toMillis(const std::chrono::system_clock::time_point& datetime) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(datetime.time_since_epoch()).count();
}

std::tm lifuren::dates::parseTm(const uint64_t& millis) {
    const auto duration  = std::chrono::milliseconds(millis);
    const auto timePoint = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(duration);
    const auto timestamp = std::chrono::system_clock::to_time_t(timePoint);
    return *std::localtime(&timestamp);
}

std::chrono::system_clock::time_point lifuren::dates::parseTp(const uint64_t& millis) {
    const auto duration  = std::chrono::milliseconds(millis);
    const auto timePoint = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(duration);
    const auto timestamp = std::chrono::system_clock::to_time_t(timePoint);
    return std::chrono::system_clock::from_time_t(timestamp);
}
