#include "lifuren/GUI.hpp"

#include "wx/wx.h"

static wxPanel   * panel      { nullptr };
static wxButton  * home_page  { nullptr };
static wxButton  * gitee_page { nullptr };
static wxButton  * github_page{ nullptr };
static wxTextCtrl* about      { nullptr };

static auto home_page_text   = wxT("主页");
static auto gitee_page_text  = wxT("Gitee");
static auto github_page_text = wxT("Github");
static auto about_text       = wxT("关于");

static const auto home_page_id   = 2000;
static const auto gitee_page_id  = 2001;
static const auto github_page_id = 2002;
static const auto about_id       = 2003;

lifuren::AboutWindow::AboutWindow(int width, int height, const wxString& title) : Window(width, height, title) {
}

lifuren::AboutWindow::~AboutWindow() {
    panel       = nullptr;
    home_page   = nullptr;
    gitee_page  = nullptr;
    github_page = nullptr;
    about       = nullptr;
}

void lifuren::AboutWindow::drawWidget() {
    const int w = this->GetClientSize().GetWidth();
    const int h = this->GetClientSize().GetHeight();
    panel       = new wxPanel(this);
    about       = new wxTextCtrl(panel, about_id,       about_text,       wxPoint(         10,      10), wxSize(w - 20, h - 80), wxTE_MULTILINE);
    home_page   = new wxButton(  panel, home_page_id,   home_page_text,   wxPoint(w / 2 - 140,  h - 50), wxSize(    80,     30));
    gitee_page  = new wxButton(  panel, gitee_page_id,  gitee_page_text,  wxPoint(w / 2 -  40,  h - 50), wxSize(    80,     30));
    github_page = new wxButton(  panel, github_page_id, github_page_text, wxPoint(w / 2 +  60,  h - 50), wxSize(    80,     30));
    about->Disable();
    about->SetBackgroundColour(panel->GetBackgroundColour());
}

void lifuren::AboutWindow::bindEvent() {
    this->Bind(wxEVT_BUTTON, [](const wxCommandEvent& event) {
        auto id = event.GetId();
        switch(id) {
            case home_page_id  : wxLaunchDefaultBrowser("https://www.acgist.com");            break;
            case gitee_page_id : wxLaunchDefaultBrowser("https://gitee.com/acgist/lifuren");  break;
            case github_page_id: wxLaunchDefaultBrowser("https://github.com/acgist/lifuren"); break;
        }
    });
}

void lifuren::AboutWindow::fillData() {
    about->Clear();
    about->AppendText(wxT(R"(李夫人
    
北方有佳人，绝世而独立。
一顾倾人城，再顾倾人国。
宁不知倾城与倾国，佳人难再得。

https://www.acgist.com
https://gitee.com/acgist/lifuren
https://github.com/acgist/lifuren

Copyright(c) 2024-present acgist. All Rights Reserved.

http://www.apache.org/licenses/LICENSE-2.0
)"));
}
