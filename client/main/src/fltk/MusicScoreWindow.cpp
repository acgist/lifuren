#include "lifuren/FLTK.hpp"

/**
 * TODO: 乐谱显示、钢琴演奏、钢琴指法、移调
 */
#include "spdlog/spdlog.h"

#include "lifuren/Raii.hpp"

lifuren::MusicScoreWindow::MusicScoreWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::MusicScoreWindow::~MusicScoreWindow() {
}

void lifuren::MusicScoreWindow::drawElement() {
}

void lifuren::MusicScoreWindow::bindEvent() {
}

void lifuren::MusicScoreWindow::fillData() {
}
