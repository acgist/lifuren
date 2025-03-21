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

static wxPanel * panel{ nullptr };
#if wxUSE_WEBVIEW
static wxWebView* web_view{ nullptr };
#endif

static void push_audio();

lifuren::MusicScoreWindow::MusicScoreWindow(int width, int height, const wxString& title) : Window(width, height, title) {
}

lifuren::MusicScoreWindow::~MusicScoreWindow() {
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
        auto file = lifuren::config::baseFile("./webview/index.html");
        SPDLOG_DEBUG("加载页面：{}", file);
        web_view->LoadURL(file);
    }
}

static void push_audio() {
    auto path = wxFileName(wxStandardPaths::Get().GetExecutablePath().BeforeLast(wxFileName::GetPathSeparator()) + wxFileName::GetPathSeparator() + "webview/audio/1.mp3");
    wxFileSystem fs;
    auto file = fs.OpenFile(path.GetFullPath());
    if(!file) {
        SPDLOG_WARN("打开文件失败：{}", path.GetFullPath().mb_str(wxConvUTF8).data());
        return;
    }
    auto stream = file->GetStream();
    if(!stream) {
        SPDLOG_WARN("打开文件失败：{}", path.GetFullPath().mb_str(wxConvUTF8).data());
        return;
    }
    size_t length = stream->GetLength();
    std::vector<char> data(length);
    stream->ReadAll(data.data(), length);
    auto str = wxBase64Encode(data.data(), length);
    web_view->RunScriptAsync("register_audio(1,`" + str + "`)");
    delete file;
}
