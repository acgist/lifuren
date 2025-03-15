#include "lifuren/FLTK.hpp"

#include <memory>
#include <thread>
#include <functional>

#include "lifuren/File.hpp"
#include "lifuren/Raii.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Message.hpp"
#include "lifuren/audio/Audio.hpp"
#include "lifuren/image/Image.hpp"

#include "FL/fl_ask.H"
#include "FL/filename.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Display.H"
#include "FL/Fl_Shared_Image.H"

static Fl_Button      * bachButtonPtr      { nullptr };
static Fl_Button      * chopinButtonPtr    { nullptr };
static Fl_Button      * mozartButtonPtr    { nullptr };
static Fl_Button      * wudaoziButtonPtr   { nullptr };
static Fl_Button      * shikuangButtonPtr  { nullptr };
static Fl_Button      * musicScoreButtonPtr{ nullptr };
static Fl_Button      * configButtonPtr    { nullptr };
static Fl_Button      * aboutButtonPtr     { nullptr };
static Fl_Text_Display* displayPtr         { nullptr };
static Fl_Text_Buffer * bufferPtr          { nullptr };

static lifuren::AboutWindow     * aboutWindowPtr     { nullptr };
static lifuren::ConfigWindow    * configWindowPtr    { nullptr };
static lifuren::MusicScoreWindow* musicScoreWindowPtr{ nullptr };

static bool running = false;
static std::shared_ptr<std::thread> thread{ nullptr };

static void bachCallback      (Fl_Widget*, void*);
static void chopinCallback    (Fl_Widget*, void*);
static void mozartCallback    (Fl_Widget*, void*);
static void wudaoziCallback   (Fl_Widget*, void*);
static void shikuangCallback  (Fl_Widget*, void*);
static void musicScoreCallback(Fl_Widget*, void*);
static void configCallback    (Fl_Widget*, void*);
static void aboutCallback     (Fl_Widget*, void*);

static void messageCallback(const char*);
static bool run(const char*, const char*, std::function<void(std::string)>);

lifuren::MainWindow::MainWindow(int width, int height, const char* title) : Window(width, height, title) {
    fl_register_images();
}

lifuren::MainWindow::~MainWindow() {
    if(thread) {
        thread->join();
        thread = nullptr;
    }
    LFR_DELETE_PTR(bachButtonPtr      );
    LFR_DELETE_PTR(chopinButtonPtr    );
    LFR_DELETE_PTR(mozartButtonPtr    );
    LFR_DELETE_PTR(wudaoziButtonPtr   );
    LFR_DELETE_PTR(shikuangButtonPtr  );
    LFR_DELETE_PTR(musicScoreButtonPtr);
    LFR_DELETE_PTR(configButtonPtr    );
    LFR_DELETE_PTR(aboutButtonPtr     );
    LFR_DELETE_PTR(displayPtr         );
    LFR_DELETE_PTR(bufferPtr          );
    lifuren::message::unregisterMessageCallback();
}

void lifuren::MainWindow::drawElement() {
    bachButtonPtr       = new Fl_Button(                           20,  10, (this->w() - 60) / 3, 80, "音频识谱");
    chopinButtonPtr     = new Fl_Button((this->w() - 60) / 3     + 30,  10, (this->w() - 60) / 3, 80, "简谱识谱");
    mozartButtonPtr     = new Fl_Button((this->w() - 60) / 3 * 2 + 40,  10, (this->w() - 60) / 3, 80, "五线谱识谱");
    shikuangButtonPtr   = new Fl_Button(                           20, 100, (this->w() - 50) / 2, 80, "音频风格迁移");
    wudaoziButtonPtr    = new Fl_Button((this->w() - 50) / 2     + 30, 100, (this->w() - 50) / 2, 80, "图片风格迁移");
    musicScoreButtonPtr = new Fl_Button(                           20, 190, (this->w() - 40)    , 80, "乐谱查看");
    configButtonPtr     = new Fl_Button(                           20, 280, (this->w() - 50) / 2, 80, "配置");
    aboutButtonPtr      = new Fl_Button((this->w() - 50) / 2     + 30, 280, (this->w() - 50) / 2, 80, "关于");
    displayPtr          = new Fl_Text_Display(20, 370, this->w() - 40, this->h() - 380);
    bufferPtr           = new Fl_Text_Buffer();
    displayPtr->begin();
    displayPtr->box(FL_NO_BOX);
    displayPtr->color(FL_BACKGROUND_COLOR);
    displayPtr->buffer(bufferPtr);
    displayPtr->wrap_mode(displayPtr->WRAP_AT_COLUMN, displayPtr->textfont());
    displayPtr->end();
}

void lifuren::MainWindow::bindEvent() {
    lifuren::message::registerMessageCallback(messageCallback);
    bachButtonPtr      ->callback(bachCallback      , this);
    chopinButtonPtr    ->callback(chopinCallback    , this);
    mozartButtonPtr    ->callback(mozartCallback    , this);
    wudaoziButtonPtr   ->callback(wudaoziCallback   , this);
    shikuangButtonPtr  ->callback(shikuangCallback  , this);
    musicScoreButtonPtr->callback(musicScoreCallback, this);
    configButtonPtr    ->callback(configCallback    , this);
    aboutButtonPtr     ->callback(aboutCallback     , this);
}

static void bachCallback(Fl_Widget*, void*) {
    run("选择音频", "*.{aac,mp3,flac}", [](std::string file) {
        auto client = lifuren::audio::getAudioClient("bach");
        const auto [success, output_file] = client->pred(file);
        if(success) {
            fl_open_uri(("file:///" + lifuren::file::parent(output_file)).c_str());
        }
    });
}

static void chopinCallback(Fl_Widget*, void*) {
    run("选择简谱", "*.{png,jpg,jpeg}", [](std::string file) {
        auto client = lifuren::audio::getAudioClient("chopin");
        const auto [success, output_file] = client->pred(file);
        if(success) {
            fl_open_uri(("file:///" + lifuren::file::parent(output_file)).c_str());
        }
    });
}

static void mozartCallback(Fl_Widget*, void*) {
    run("选择五线谱", "*.{png,jpg,jpeg}", [](std::string file) {
        auto client = lifuren::audio::getAudioClient("mozart");
        const auto [success, output_file] = client->pred(file);
        if(success) {
            fl_open_uri(("file:///" + lifuren::file::parent(output_file)).c_str());
        }
    });
}

static void wudaoziCallback(Fl_Widget*, void*) {
    run("选择音频", "*.{aac,mp3,flac}", [](std::string file) {
        auto client = lifuren::audio::getAudioClient("wudaozi");
        const auto [success, output_file] = client->pred(file);
        if(success) {
            fl_open_uri(("file:///" + lifuren::file::parent(output_file)).c_str());
        }
    });
}

static void shikuangCallback(Fl_Widget*, void*) {
    run("选择图片", "*.{png,jpg,jpeg}", [](std::string file) {
        auto client = lifuren::audio::getAudioClient("shikuang");
        const auto [success, output_file] = client->pred(file);
        if(success) {
            fl_open_uri(("file:///" + lifuren::file::parent(output_file)).c_str());
        }
    });
}

static void musicScoreCallback(Fl_Widget*, void* voidPtr) {
    bufferPtr->text("");
    if(musicScoreWindowPtr != nullptr) {
        musicScoreWindowPtr->show();
        return;
    }
    musicScoreWindowPtr = new lifuren::MusicScoreWindow(LFR_DIALOG_WIDTH, LFR_DIALOG_HEIGHT);
    musicScoreWindowPtr->init();
    musicScoreWindowPtr->show();
    musicScoreWindowPtr->callback([](Fl_Widget* widgetPtr, void*) -> void {
        widgetPtr->hide();
        LFR_DELETE_PTR(musicScoreWindowPtr);
    }, voidPtr);
}

static void configCallback(Fl_Widget*, void* voidPtr) {
    bufferPtr->text("");
    if(configWindowPtr != nullptr) {
        configWindowPtr->show();
        return;
    }
    configWindowPtr = new lifuren::ConfigWindow(LFR_DIALOG_WIDTH, LFR_DIALOG_HEIGHT);
    configWindowPtr->init();
    configWindowPtr->show();
    configWindowPtr->callback([](Fl_Widget* widgetPtr, void*) -> void {
        widgetPtr->hide();
        LFR_DELETE_PTR(configWindowPtr);
    }, voidPtr);
}

static void aboutCallback(Fl_Widget*, void* voidPtr) {
    bufferPtr->text("");
    if(aboutWindowPtr != nullptr) {
        aboutWindowPtr->show();
        return;
    }
    aboutWindowPtr = new lifuren::AboutWindow(LFR_DIALOG_WIDTH, LFR_DIALOG_HEIGHT);
    aboutWindowPtr->init();
    aboutWindowPtr->show();
    aboutWindowPtr->callback([](Fl_Widget* widgetPtr, void*) -> void {
        widgetPtr->hide();
        LFR_DELETE_PTR(aboutWindowPtr);
    }, voidPtr);
}

static void messageCallback(const char* message) {
    bufferPtr->append(message);
}

static bool run(const char* title, const char* filter, std::function<void(std::string)> fun) {
    if(running) {
        fl_message("已有任务正在运行");
        return false;
    }
    if(thread) {
        thread->join();
        thread = nullptr;
    }
    bufferPtr->text("");
    auto file = lifuren::fileChooser(title, filter);
    if(file.empty()) {
        fl_message("请%s", title);
        return false;
    }
    running = true;
    thread = std::make_shared<std::thread>(([fun, file] {
        fun(file);
        running = false;
    }));
    return true;
}
