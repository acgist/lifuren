#include "../header/String.hpp"

namespace lifuren {

namespace string {

    void format(std::string& format, const std::string* args, int length) {
        for (int index = 0; index < length; index++) {
            const size_t jndex = format.find(FLAG);
            if(jndex == std::string::npos) {
                return;
            }
            std::string arg = args[index];
            format.replace(jndex, FLAG_LENGTH, arg, 0, arg.length());
        }
    }

}

}