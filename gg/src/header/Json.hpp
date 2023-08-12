#pragma once

#include <string>

#include "nlohmann/json.hpp"

namespace lifuren {

namespace gg {
    
    /**
     * 数组->JSON
     * 
     * @param vlaues 数组
     * @param length 长度
     * 
     * @return JSON
     */
    template<typename T>
    extern std::string toJSON(const T* values, int length = 0);

    template<typename T>
    std::string toJSON(const T* values, int length) {
        nlohmann::json array = nlohmann::json::array();
        for(int index = 0; index < length; index++) {
            array.push_back(values[index]);
        }
        return array.dump();
    }

}

}