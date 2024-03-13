#include "../../header/utils/Dates.hpp"

std::string lifuren::dates::format(const std::tm& datetime, const std::string& format) {
    std::ostringstream output;
    output << std::put_time(&datetime, format.c_str());
    return output.str();
}

std::string lifuren::dates::format(const std::chrono::system_clock::time_point& datetime, const std::string& format) {
    const std::time_t timestamp = std::chrono::system_clock::to_time_t(datetime);
    const std::tm tm = *std::gmtime(&timestamp);
    // const std::tm tm = *std::localtime(&timestamp);
    return lifuren::dates::format(tm, format);
}

std::tm lifuren::dates::parseTm(const std::string& datetime, const std::string& format) {
    std::tm tm;
    std::istringstream input(datetime);
	input >> std::get_time(&tm, format.c_str());
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

std::tm lifuren::dates::toDatetimeTm(const uint64_t& millis) {
    const auto duration = std::chrono::milliseconds(millis);
    const auto timePoint = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(duration);
    const auto timestamp = std::chrono::system_clock::to_time_t(timePoint);
    return *std::gmtime(&timestamp);
    // return *std::localtime(&timestamp);
}

std::chrono::system_clock::time_point lifuren::dates::toDatetimeTp(const uint64_t& millis) {
    const auto duration = std::chrono::milliseconds(millis);
    const auto timePoint = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(duration);
    const auto timestamp = std::chrono::system_clock::to_time_t(timePoint);
    return std::chrono::system_clock::from_time_t(timestamp);
}
