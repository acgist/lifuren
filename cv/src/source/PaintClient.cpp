#include "lifuren/PaintClient.hpp"

lifuren::PaintClient::PaintClient(lifuren::PaintClient::PaintCallback callback) : callback(callback) {
}

lifuren::PaintClient::~PaintClient() {
}
