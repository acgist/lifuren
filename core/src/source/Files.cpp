#include "../header/Files.hpp"

void lifuren::files::listFiles(std::vector<std::string>& vector, const std::string& path) {
    listFiles(vector, path, {});
}

void lifuren::files::listFiles(std::vector<std::string>& vector, const std::string& path, const std::vector<std::string>& exts) {
    listFiles(vector, path, [&](const std::string& filename) -> bool {
        if(exts.empty()) {
            return true;
        } else {
            const size_t index = filename.find_last_of('.');
            if(index == std::string::npos) {
                return false;
            }
            std::string ext = filename.substr(index);
            lifuren::strings::toLower(ext);
            const auto ret = std::find(exts.begin(), exts.end(), ext);
            return ret != exts.end();
        }
    });
}

template <typename Predicate>
void lifuren::files::listFiles(std::vector<std::string>& vector, const std::string& path, const Predicate& predicate) {
    namespace fs = std::filesystem;
    if(!fs::exists(path) || !fs::is_directory(path)) {
        SPDLOG_DEBUG("目录无效：{} - {}", __func__, path);
        return;
    }
    auto iterator = fs::directory_iterator(fs::u8path(path));
    for(const auto& entry : iterator) {
        std::string filepath = entry.path().u8string();
        if(entry.is_regular_file()) {
            std::string filename = entry.path().filename().u8string();
            if(predicate(filename)) {
                vector.push_back(filepath);
            } else {
                SPDLOG_DEBUG("忽略无效文件类型：{} - {}", __func__, filepath);
            }
        } else {
            SPDLOG_DEBUG("忽略无效文件：{} - {}", __func__, filepath);
        }
    }
}

std::string lifuren::files::loadFile(const std::string& path) {
    std::ifstream input;
    input.open(path, std::ios_base::in);
    if(!input.is_open()) {
        SPDLOG_WARN("打开文件失败：{} - {}", __func__, path);
        return "";
    }
    std::string line;
    std::string settings;
    while(std::getline(input, line)) {
        settings += line;
    }
    input.close();
    return settings;
}

void lifuren::files::saveFile(const std::string& path, const std::string& value) {
    std::ofstream output;
    output.open(path, std::ios_base::out | std::ios_base::trunc);
    if(!output.is_open()) {
        SPDLOG_WARN("打开文件失败：{}", path);
        return;
    }
    output << value;
    output.close();
}
