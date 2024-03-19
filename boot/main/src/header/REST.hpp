/**
 * REST API
 * 
 * @author acgist
 */
#pragma once

#include "httplib.h"

#include "Logger.hpp"

namespace lifuren {

extern httplib::Server httpServer;

extern void initHttpServer();

extern void shutdownHttpServer();

extern void restGetIndex();

}