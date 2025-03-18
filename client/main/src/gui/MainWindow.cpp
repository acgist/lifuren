#include "lifuren/GUI.hpp"

#include <memory>
#include <thread>
#include <functional>

#include "wx/wx.h"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Raii.hpp"
#include "lifuren/Audio.hpp"
#include "lifuren/Image.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Message.hpp"

static wxPanel   * panel             { nullptr };
static wxButton  * bach_button       { nullptr };
static wxButton  * chopin_button     { nullptr };
static wxButton  * mozart_button     { nullptr };
static wxButton  * wudaozi_button    { nullptr };
static wxButton  * shikuang_button   { nullptr };
static wxButton  * music_score_button{ nullptr };
static wxButton  * config_button     { nullptr };
static wxButton  * about_button      { nullptr };
static wxTextCtrl* message_text      { nullptr };

static void bach_callback       (const wxCommandEvent&);
static void chopin_callback     (const wxCommandEvent&);
static void mozart_callback     (const wxCommandEvent&);
static void wudaozi_callback    (const wxCommandEvent&);
static void shikuang_callback   (const wxCommandEvent&);
static void music_score_callback(const wxCommandEvent&);
static void config_callback     (const wxCommandEvent&);
static void about_callback      (const wxCommandEvent&);
static void message_callback    (const char*);

static auto bach_text         = wxT("音频识谱");
static auto chopin_text       = wxT("简谱识谱");
static auto mozart_text       = wxT("五线谱识谱");
static auto wudaozi_text      = wxT("图片风格迁移");
static auto shikuang_text     = wxT("音频风格迁移");
static auto music_score_text  = wxT("乐谱查看");
static auto config_text       = wxT("配置");
static auto about_text        = wxT("关于");
static auto message_text_text = wxT("日志");

static const auto bach_id         = 1000;
static const auto chopin_id       = 1001;
static const auto mozart_id       = 1002;
static const auto wudaozi_id      = 1004;
static const auto shikuang_id     = 1003;
static const auto music_score_id  = 1005;
static const auto config_id       = 1006;
static const auto about_id        = 1007;
static const auto message_text_id = 1008;

static int thread_event_thread  = 100;
static int thread_event_message = 101;

static bool running = false;
static lifuren::MainWindow* mainWindow{ nullptr };
static std::shared_ptr<std::thread> thread{ nullptr };

static bool run(const char*, const wxString&, const wxString&, std::function<std::tuple<bool, std::string>(std::string)>);

lifuren::MainWindow::MainWindow(int width, int height, const wxString& title) : Window(width, height, title) {
    mainWindow = this;
}

lifuren::MainWindow::~MainWindow() {
    if(thread) {
        thread->join();
        thread = nullptr;
    }
    lifuren::message::unregisterMessageCallback();
    mainWindow = nullptr;
}

void lifuren::MainWindow::drawElement() {
    panel              = new wxPanel(this);
    bach_button        = new wxButton  (panel, bach_id,         bach_text,         wxPoint( 10,  10), wxSize( 400,  80));
    chopin_button      = new wxButton  (panel, chopin_id,       chopin_text,       wxPoint(420,  10), wxSize( 400,  80));
    mozart_button      = new wxButton  (panel, mozart_id,       mozart_text,       wxPoint(830,  10), wxSize( 400,  80));
    shikuang_button    = new wxButton  (panel, shikuang_id,     shikuang_text,     wxPoint( 10, 100), wxSize( 605,  80));
    wudaozi_button     = new wxButton  (panel, wudaozi_id,      wudaozi_text,      wxPoint(625, 100), wxSize( 605,  80));
    music_score_button = new wxButton  (panel, music_score_id,  music_score_text,  wxPoint( 10, 190), wxSize(1220,  80));
    config_button      = new wxButton  (panel, config_id,       config_text,       wxPoint( 10, 280), wxSize( 605,  80));
    about_button       = new wxButton  (panel, about_id,        about_text,        wxPoint(625, 280), wxSize( 605,  80));
    message_text       = new wxTextCtrl(panel, message_text_id, message_text_text, wxPoint( 10, 370), wxSize(1220, 300), wxTE_MULTILINE);
    message_text->Disable();
    message_text->SetBackgroundColour(panel->GetBackgroundColour());
}

void lifuren::MainWindow::bindEvent() {
    this->Bind(wxEVT_THREAD, [](const wxThreadEvent& event) {
        if(event.GetInt() == thread_event_thread) {
            auto pref = wxT("explorer file:///");
            auto path = event.GetString();
            wxExecute(pref + path, wxEXEC_ASYNC, nullptr);
        } else if(event.GetInt() == thread_event_message) {
            message_text->AppendText(event.GetString());
            message_text->Refresh();
        } else {
            // -
        }
    });
    this->Bind(wxEVT_BUTTON, [](const wxCommandEvent& event) {
        const auto id = event.GetId();
        switch(id) {
            case bach_id       : bach_callback(event);        break;
            case chopin_id     : chopin_callback(event);      break;
            case mozart_id     : mozart_callback(event);      break;
            case wudaozi_id    : wudaozi_callback(event);     break;
            case shikuang_id   : shikuang_callback(event);    break;
            case music_score_id: music_score_callback(event); break;
            case config_id     : config_callback(event);      break;
            case about_id      : about_callback(event);       break;
        }
    });
    lifuren::message::registerMessageCallback(message_callback);
}

static void bach_callback(const wxCommandEvent&) {
    run("音频识谱", wxT("选择音频"), wxT("音频文件|*.aac;*.mp3;*.flac"), [](std::string file) -> std::tuple<bool, std::string> {
        auto client = lifuren::audio::getAudioClient("bach");
        if(client->load(lifuren::config::CONFIG.model_bach)) {
            return client->pred(file);
        } else {
            return { false, {} };
        }
    });
}

static void chopin_callback(const wxCommandEvent&) {
    run("简谱识谱", wxT("选择简谱"), wxT("图片文件|*.png;*.jpg;*.jpeg"), [](std::string file) -> std::tuple<bool, std::string> {
        auto client = lifuren::image::getImageClient("chopin");
        if(client->load(lifuren::config::CONFIG.model_chopin)) {
            return client->pred(file);
        } else {
            return { false, {} };
        }
    });
}

static void mozart_callback(const wxCommandEvent&) {
    run("五线谱识谱", wxT("选择五线谱"), wxT("图片文件|*.png;*.jpg;*.jpeg"), [](std::string file) -> std::tuple<bool, std::string> {
        auto client = lifuren::image::getImageClient("mozart");
        if(client->load(lifuren::config::CONFIG.model_mozart)) {
            return client->pred(file);
        } else {
            return { false, {} };
        }
    });
}

static void wudaozi_callback(const wxCommandEvent&) {
    run("图片风格迁移", wxT("选择图片"), wxT("图片文件|*.png;*.jpg;*.jpeg"), [](std::string file) -> std::tuple<bool, std::string> {
        auto client = lifuren::image::getImageClient("wudaozi");
        if(client->load(lifuren::config::CONFIG.model_wudaozi)) {
            return client->pred(file);
        } else {
            return { false, {} };
        }
    });
}

static void shikuang_callback(const wxCommandEvent&) {
    run("音频风格迁移", wxT("选择音频"), wxT("音频文件|*.aac;*.mp3;*.flac"), [](std::string file) -> std::tuple<bool, std::string> {
        auto client = lifuren::audio::getAudioClient("shikuang");
        if(client->load(lifuren::config::CONFIG.model_shikuang)) {
            return client->pred(file);
        } else {
            return { false, {} };
        }
    });
}

static void music_score_callback(const wxCommandEvent&) {
    message_text->Clear();
    lifuren::MusicScoreWindow* musicScoreWindow = new lifuren::MusicScoreWindow(LFR_DIALOG_WIDTH, LFR_DIALOG_HEIGHT);
    musicScoreWindow->init();
    musicScoreWindow->Show(true);
    musicScoreWindow->Maximize(true);
}

static void config_callback(const wxCommandEvent&) {
    message_text->Clear();
    lifuren::ConfigWindow* configWindow = new lifuren::ConfigWindow(LFR_DIALOG_WIDTH, LFR_DIALOG_HEIGHT);
    configWindow->init();
    configWindow->Show(true);
}

static void about_callback(const wxCommandEvent&) {
    message_text->Clear();
    lifuren::AboutWindow* aboutWindow = new lifuren::AboutWindow(LFR_DIALOG_WIDTH, LFR_DIALOG_HEIGHT);
    aboutWindow->init();
    aboutWindow->Show(true);
}

static void message_callback(const char* message) {
    auto event = new wxThreadEvent();
    event->SetInt(thread_event_message);
    event->SetString(wxString::FromUTF8(message));
    wxQueueEvent(mainWindow, event);
}

static bool run(const char* name, const wxString& title, const wxString& filter, std::function<std::tuple<bool, std::string>(std::string)> fun) {
    if(running) {
        wxMessageDialog dialog(
            nullptr,
            wxT("已有任务正在运行"),
            wxT("消息提示"),
            wxOK | wxCENTRE | wxICON_WARNING
        );
        dialog.ShowModal();
        return false;
    }
    if(thread) {
        thread->join();
        thread = nullptr;
    }
    message_text->Clear();
    auto file = lifuren::file_chooser(title, filter);
    if(file.empty()) {
        wxMessageDialog dialog(
            nullptr,
            wxT("需要选择一个文件"),
            wxT("消息提示"),
            wxOK | wxCENTRE | wxICON_WARNING
        );
        dialog.ShowModal();
        return false;
    }
    running = true;
    thread = std::make_shared<std::thread>(([fun, name, file, title] {
        SPDLOG_DEBUG("开始任务：{}", name);
        try {
            auto [success, path] = fun(file);
            if(success) {
                SPDLOG_DEBUG("任务完成：{}", name);
                auto event = new wxThreadEvent();
                event->SetInt(thread_event_thread);
                event->SetString(wxString::FromUTF8(lifuren::file::parent(path)));
                wxQueueEvent(mainWindow, event);
            } else {
                SPDLOG_WARN("任务失败：{}", name);
            }
        } catch(const std::exception& e) {
            SPDLOG_ERROR("任务异常：{}", name);
        } catch(...) {
            SPDLOG_ERROR("任务异常：{}", name);
        }
        running = false;
    }));
    return true;
}
