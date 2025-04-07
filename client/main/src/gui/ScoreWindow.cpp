#include "lifuren/GUI.hpp"

#include "wx/wx.h"
#include "wx/base64.h"
#include "wx/filesys.h"
#include "wx/webview.h"
#include <wx/filename.h>
#include <wx/stdpaths.h>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"

static wxPanel  * panel   { nullptr };
static wxWebView* web_view{ nullptr };

#if defined(_DEBUG) || !defined(NDEBUG)
const static wxString debug_path = wxT("D:/gitee/lifuren/client/");
#endif

static void push_audio();

lifuren::MusicScoreWindow::MusicScoreWindow(int width, int height, const wxString& title) : Window(width, height, title) {
}

lifuren::MusicScoreWindow::~MusicScoreWindow() {
    // 置空
    panel    = nullptr;
    web_view = nullptr;
}

void lifuren::MusicScoreWindow::drawElement() {
    const int w = wxSystemSettings::GetMetric(wxSYS_SCREEN_X, nullptr);
    const int h = wxSystemSettings::GetMetric(wxSYS_SCREEN_Y, nullptr);
    panel = new wxPanel(this);
    if(wxWebView::IsBackendAvailable(wxWebViewBackendEdge)) {
        web_view = wxWebView::New(panel, wxID_ANY, wxWebViewDefaultURLStr, wxPoint(0, 0), wxSize(w, h), wxWebViewBackendEdge);
    } else if(wxWebView::IsBackendAvailable(wxWebViewBackendWebKit)) {
        web_view = wxWebView::New(panel, wxID_ANY, wxWebViewDefaultURLStr, wxPoint(0, 0), wxSize(w, h), wxWebViewBackendWebKit);
    } else if(wxWebView::IsBackendAvailable(wxWebViewBackendDefault)) {
        web_view = wxWebView::New(panel, wxID_ANY, wxWebViewDefaultURLStr, wxPoint(0, 0), wxSize(w, h), wxWebViewBackendDefault);
    } else {
        SPDLOG_DEBUG("没有适配的浏览器内核");
    }
    if(web_view) {
        #if defined(_DEBUG) || !defined(NDEBUG)
        web_view->EnableAccessToDevTools(true);
        #else
        web_view->EnableAccessToDevTools(false);
        #endif
    }
}

void lifuren::MusicScoreWindow::bindEvent() {
    web_view->AddScriptMessageHandler("lfr_backend");
    web_view->Bind(wxEVT_WEBVIEW_SCRIPT_MESSAGE_RECEIVED, [](const wxWebViewEvent& event) {
        auto command = event.GetString();
        if(command == wxString(wxT("audio"))) {
            push_audio();
        } else {
            SPDLOG_WARN("没有适配命令：{}", command.mb_str(wxConvUTF8).data());
        }
    });
}

void lifuren::MusicScoreWindow::fillData() {
    if(web_view) {
        #if defined(_DEBUG) || !defined(NDEBUG)
        auto file = debug_path + wxString(wxT("webview/index.html"));
        #else
        auto file = lifuren::app_base_dir("webview/index.html");
        #endif
        SPDLOG_DEBUG("加载页面：{}", file.mb_str(wxConvUTF8).data());
        web_view->LoadURL(file);
    }
}

static void push_audio() {
    wxFileSystem fs;
    for(int i = 1; i <= 88; ++i) {
        if(!web_view) {
            return;
        }
        #if defined(_DEBUG) || !defined(NDEBUG)
        auto path = debug_path + wxT("webview/audio/" + std::to_string(i) + ".mp3");
        #else
        auto path = lifuren::app_base_dir(wxT("webview/audio/" + std::to_string(i) + ".mp3"));
        #endif
        auto file = fs.OpenFile(path);
        if(!file) {
            SPDLOG_WARN("打开文件失败：{}", path.mb_str(wxConvUTF8).data());
            return;
        }
        auto stream = file->GetStream();
        if(!stream) {
            SPDLOG_WARN("打开文件失败：{}", path.mb_str(wxConvUTF8).data());
            return;
        }
        size_t length = stream->GetLength();
        std::vector<char> data(length);
        stream->ReadAll(data.data(), length);
        auto audio_source = wxBase64Encode(data.data(), length);
        web_view->RunScript(wxT("lifuren.register_audio_source('" + std::to_string(i) + "','piano', `" + audio_source + "`)"));
        SPDLOG_DEBUG("注册音源：{} - {}", i, length);
        delete file;
    }
}
