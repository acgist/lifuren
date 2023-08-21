#include "../header/String.hpp"

void lifuren::format(std::string& format, std::string& flag, const std::string* args, int length) {
    for (int index = 0; index < length; index++) {
        const size_t jndex = format.find(flag);
        if(jndex == std::string::npos) {
            return;
        }
        std::string arg = args[index];
        format.replace(jndex, flag.size(), arg, 0, arg.length());
    }
}