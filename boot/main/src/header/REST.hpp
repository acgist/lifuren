/**
 * REST API
 * 
 * REST.hpp必须在FLTK.hpp的前面
 * 
 * @author acgist
 */
#pragma once

#include "httplib.h"

#include "nlohmann/json.hpp"

#include "Logger.hpp"

namespace lifuren {

extern httplib::Server httpServer;

extern void initHttpServer();

extern void shutdownHttpServer();

extern nlohmann::json buildResponse(const char* code, const char* message);

extern void restGetIndex();

extern void restGetShutdown();

}