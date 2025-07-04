#include "lifuren/GUI.hpp"

#include <memory>
#include <thread>
#include <functional>

#include "wx/wx.h"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"
#include "lifuren/Message.hpp"
#include "lifuren/Wudaozi.hpp"

static wxPanel   * panel        { nullptr };
static wxButton  * image_button { nullptr };
static wxButton  * video_button { nullptr };
static wxButton  * config_button{ nullptr };
static wxButton  * about_button { nullptr };
static wxTextCtrl* message_ctrl { nullptr };

static void image_callback  (const wxCommandEvent&);
static void video_callback  (const wxCommandEvent&);
static void config_callback (const wxCommandEvent&);
static void about_callback  (const wxCommandEvent&);
static void message_callback(const char*);

static const auto image_text   = wxT("图片生成");
static const auto video_text   = wxT("视频生成");
static const auto config_text  = wxT("配置");
static const auto about_text   = wxT("关于");
static const auto message_text = wxT("日志");

static const auto image_id   = 1000;
static const auto video_id   = 1001;
static const auto config_id  = 1002;
static const auto about_id   = 1003;
static const auto message_id = 1004;

static const int thread_event_thread  = 100;
static const int thread_event_message = 101;

static bool running = false;

static lifuren::MainWindow* mainWindow{ nullptr };

static std::shared_ptr<std::thread> thread{ nullptr };

static bool run(bool file_choose, const char*, const wxString&, const wxString&, std::function<std::tuple<bool, std::string>(std::string)>);

lifuren::MainWindow::MainWindow(int width, int height, const wxString& title) : Window(width, height, title) {
    mainWindow = this;
}

lifuren::MainWindow::~MainWindow() {
    if(thread) {
        thread->join();
        thread = nullptr;
    }
    lifuren::message::unregister_message_callback();
    mainWindow    = nullptr;
    panel         = nullptr;
    image_button  = nullptr;
    video_button  = nullptr;
    config_button = nullptr;
    about_button  = nullptr;
    message_ctrl  = nullptr;
}

void lifuren::MainWindow::drawWidget() {
    const int w = this->GetClientSize().GetWidth();
    const int h = this->GetClientSize().GetHeight();
    panel         = new wxPanel(this);
    image_button  = new wxButton  (panel, image_id,   image_text,   wxPoint(          10,  10), wxSize((w - 20),          80));
    video_button  = new wxButton  (panel, video_id,   video_text,   wxPoint(          10, 100), wxSize((w - 20),          80));
    config_button = new wxButton  (panel, config_id,  config_text,  wxPoint(          10, 190), wxSize((w - 30) / 2,      80));
    about_button  = new wxButton  (panel, about_id,   about_text,   wxPoint((w / 2) +  5, 190), wxSize((w - 30) / 2,      80));
    message_ctrl  = new wxTextCtrl(panel, message_id, message_text, wxPoint(          10, 270), wxSize((w - 20),     h - 280), wxTE_MULTILINE);
    message_ctrl->Disable();
    message_ctrl->SetBackgroundColour(panel->GetBackgroundColour());
}

void lifuren::MainWindow::bindEvent() {
    this->Bind(wxEVT_THREAD, [](const wxThreadEvent& event) {
        if(event.GetInt() == thread_event_thread) {
            auto success = event.GetPayload<bool>();
            if(success) {
                auto pref = wxT("explorer file:///");
                auto path = event.GetString();
                wxExecute(pref + path, wxEXEC_ASYNC, nullptr);
            } else {
                wxMessageDialog dialog(
                    nullptr,
                    wxT("任务执行失败"),
                    wxT("失败提示"),
                    wxOK | wxCENTRE | wxICON_ERROR
                );
                dialog.ShowModal();
            }
        } else if(event.GetInt() == thread_event_message) {
            message_ctrl->AppendText(event.GetString());
            message_ctrl->Refresh();
        } else {
            // -
        }
    });
    this->Bind(wxEVT_BUTTON, [](const wxCommandEvent& event) {
        const auto id = event.GetId();
        switch(id) {
            case image_id  : image_callback(event);  break;
            case video_id  : video_callback(event);  break;
            case config_id : config_callback(event); break;
            case about_id  : about_callback(event);  break;
        }
    });
    lifuren::message::register_message_callback(message_callback);
}

static void image_callback(const wxCommandEvent&) {
    run(false, "图片生成", wxT("保存目录"), wxT(""), [](std::string file) -> std::tuple<bool, std::string> {
        auto client = lifuren::get_wudaozi_client();
        if(client->load(lifuren::config::CONFIG.model_wudaozi)) {
            return client->pred(file);
        } else {
            return { false, {} };
        }
    });
}

static void video_callback(const wxCommandEvent&) {
    run(true, "视频生成", wxT("选择媒体"), wxT("媒体文件|*.png;*.jpg;*.jpeg;*.mp4"), [](std::string file) -> std::tuple<bool, std::string> {
        auto client = lifuren::get_wudaozi_client();
        if(client->load(lifuren::config::CONFIG.model_wudaozi)) {
            return client->pred(file);
        } else {
            return { false, {} };
        }
    });
}

static void config_callback(const wxCommandEvent&) {
    message_ctrl->Clear();
    lifuren::ConfigWindow* configWindow = new lifuren::ConfigWindow(LFR_DIALOG_WIDTH, LFR_DIALOG_HEIGHT);
    configWindow->init();
    configWindow->Show(true);
}

static void about_callback(const wxCommandEvent&) {
    message_ctrl->Clear();
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

static bool run(bool file_choose, const char* name, const wxString& title, const wxString& filter, std::function<std::tuple<bool, std::string>(std::string)> fun) {
    if(running) {
        wxMessageDialog dialog(
            nullptr,
            wxT("已有任务正在运行"),
            wxT("失败提示"),
            wxOK | wxCENTRE | wxICON_WARNING
        );
        dialog.ShowModal();
        return false;
    }
    if(thread) {
        thread->join();
        thread = nullptr;
    }
    message_ctrl->Clear();
    std::string file;
    if(file_choose) {
        file = lifuren::file_chooser(title, filter);
        if(file.empty()) {
            wxMessageDialog dialog(
                nullptr,
                wxT("需要选择一个文件"),
                wxT("失败提示"),
                wxOK | wxCENTRE | wxICON_WARNING
            );
            dialog.ShowModal();
            return false;
        }
    } else {
        file = lifuren::directory_chooser(title);
        if(file.empty()) {
            wxMessageDialog dialog(
                nullptr,
                wxT("需要选择一个目录"),
                wxT("失败提示"),
                wxOK | wxCENTRE | wxICON_WARNING
            );
            dialog.ShowModal();
            return false;
        }
    }
    running = true;
    thread = std::make_shared<std::thread>(([fun, name, file, title] {
        SPDLOG_DEBUG("开始任务：{}", name);
        auto event = new wxThreadEvent();
        event->SetInt(thread_event_thread);
        try {
            auto [success, path] = fun(file);
            if(success) {
                SPDLOG_DEBUG("任务完成：{}", name);
                event->SetString(wxString::FromUTF8(lifuren::file::parent(path)));
                event->SetPayload<bool>(true);
            } else {
                SPDLOG_WARN("任务失败：{}", name);
                event->SetPayload<bool>(false);
            }
        } catch(const std::exception& e) {
            SPDLOG_ERROR("任务异常：{}", name);
            event->SetPayload<bool>(false);
        } catch(...) {
            SPDLOG_ERROR("任务异常：{}", name);
            event->SetPayload<bool>(false);
        }
        wxQueueEvent(mainWindow, event);
        running = false;
    }));
    return true;
}
