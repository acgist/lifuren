/**
 * 窗口定义
 * 
 * @author acgist
 */
#pragma once

namespace lifuren {

/**
 * 抽象窗口
 */
class LFRWindow {

};

/**
 * 主窗口
 */
class MainWindow : public LFRWindow {

public:
    /**
     * 关于
     */
    void about();
    /**
     * 设置
     */
    void setting();

};

/**
 * 设置窗口
 */
class SettingWindow : public LFRWindow {

};

/**
 * @see AudioGC
 */
class AudioGCWindow : public LFRWindow {

};

/**
 * @see AudioTS
 */
class AudioTSWindow : public LFRWindow {

};

/**
 * @see ImageGC
 */
class ImageGCWindow : public LFRWindow {

};

/**
 * @see ImageTS
 */
class ImageTSWindow : public LFRWindow {

};

/**
 * @see PoetryGC
 */
class PoetryGCWindow : public LFRWindow {

};

/**
 * @see PoetryTS
 */
class PoetryTSWindow : public LFRWindow {

};

/**
 * @see VideoGC
 */
class VideoGCWindow : public LFRWindow {

};

/**
 * @see VideoTS
 */
class VideoTSWindow : public LFRWindow {

};

}
