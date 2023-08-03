#include "../header/ECharts.hpp"

namespace lifuren {

namespace echarts {

    void writeLineSimple(const std::string xAxis[], const double series[], const std::string type) {
        std::string format = R"(
        option = {
            xAxis: {
                type: "category",
                data: {}
            },
            yAxis: {
                type: "value"
            },
            series: [{
                data: {},
                type: "{}"
            }]
        };
        )";
        const std::string array[] = {
            lifuren::json::toJSON(xAxis,  7),
            lifuren::json::toJSON(series, 7),
            type
        };
        lifuren::string::format(format, array, 3);
        LOG(INFO) << format;
    }

}

}