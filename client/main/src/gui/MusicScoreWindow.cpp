#include "lifuren/GUI.hpp"

#include "wx/wx.h"
#include "wx/webview.h"

#include "spdlog/spdlog.h"

#include "lifuren/Config.hpp"

static wxPanel * panel{ nullptr };
#if wxUSE_WEBVIEW
static wxWebView* web_view{ nullptr };
#endif

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
        // web_view->Bind(wxEVT_WEBVIEW_SCRIPT_MESSAGE_RECEIVED, [](const wxWebViewEvent& event) {
        //     wxLogMessage("Script message received; value = %s, handler = %s", event.GetString(), event.GetMessageHandler());
        // });
    }
}

void lifuren::MusicScoreWindow::bindEvent() {
}

void lifuren::MusicScoreWindow::fillData() {
    if(web_view) {
        auto file = lifuren::config::baseFile("./webview/index.html");
        SPDLOG_DEBUG("加载页面：{}", file);
        web_view->LoadURL(file);
    }
}
