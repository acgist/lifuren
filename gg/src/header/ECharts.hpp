#pragma once

#include <string>
#include <iostream>

#include "Json.hpp"
#include "GLog.hpp"
#include "String.hpp"

namespace lifuren {

/**
 * ECharts
 */
namespace echarts {

    /**
     * https://echarts.apache.org/examples/zh/editor.html?c=line-simple
     */
    extern void writeLineSimple(const std::string xAxis[], const double series[], const std::string type = "line");
    
}

}
