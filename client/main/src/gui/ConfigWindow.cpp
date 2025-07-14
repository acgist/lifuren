#include "lifuren/GUI.hpp"

#include "wx/wx.h"

#include "spdlog/spdlog.h"

#include "lifuren/Config.hpp"

static wxPanel   * panel         { nullptr };
static wxTextCtrl* tmp_input     { nullptr };
static wxButton  * tmp_button    { nullptr };
static wxTextCtrl* wudaozi_input { nullptr };
static wxButton  * wudaozi_button{ nullptr };

static const int tmp_button_id     = 3000;
static const int wudaozi_button_id = 3001;

static const auto tmp_input_text      = wxT("临时目录");
static const auto tmp_button_text     = wxT("选择临时目录");
static const auto wudaozi_input_text  = wxT("视频生成模型文件");
static const auto wudaozi_button_text = wxT("选择视频生成模型文件");

static void chooseFileCallback     (const wxCommandEvent&, wxTextCtrl*);
static void chooseDirectoryCallback(const wxCommandEvent&, wxTextCtrl*);

lifuren::ConfigWindow::ConfigWindow(int width, int height, const wxString& title) : Window(width, height, title) {
}

lifuren::ConfigWindow::~ConfigWindow() {
    lifuren::config::CONFIG.saveFile();
    panel          = nullptr;
    tmp_input      = nullptr;
    tmp_button     = nullptr;
    wudaozi_input  = nullptr;
    wudaozi_button = nullptr;
}

void lifuren::ConfigWindow::drawWidget() {
    const int w = this->GetClientSize().GetWidth();
    panel          = new wxPanel(this);
    tmp_input      = new wxTextCtrl(panel, wxID_ANY,          "",                  wxPoint((w - 640) / 2,        10), wxSize(400, 30));
    tmp_button     = new wxButton  (panel, tmp_button_id,     tmp_button_text,     wxPoint((w - 640) / 2 + 410,  10), wxSize(240, 30));
    wudaozi_input  = new wxTextCtrl(panel, wxID_ANY,          "",                  wxPoint((w - 640) / 2,        50), wxSize(400, 30));
    wudaozi_button = new wxButton  (panel, wudaozi_button_id, wudaozi_button_text, wxPoint((w - 640) / 2 + 410,  50), wxSize(240, 30));
}

void lifuren::ConfigWindow::bindEvent() {
    this->Bind(wxEVT_BUTTON, [](const wxCommandEvent& event) {
        const auto id = event.GetId();
        switch(id) {
            case tmp_button_id     : chooseDirectoryCallback(event, tmp_input); break;
            case wudaozi_button_id : chooseFileCallback(event, wudaozi_input);  break;
        }
    });
}

void lifuren::ConfigWindow::fillData() {
    const auto& config = lifuren::config::CONFIG;
    tmp_input    ->AppendText(wxString::FromUTF8(config.tmp));
    wudaozi_input->AppendText(wxString::FromUTF8(config.wudaozi));
}

static void chooseFileCallback(const wxCommandEvent&, wxTextCtrl* input) {
    auto file = lifuren::file_chooser(wxT("选择模型"), wxT("模型文件|*.pt;*.pth"));
    if(file.empty()) {
        return;
    }
    input->Clear();
    input->AppendText(wxString::FromUTF8(file));
    auto& config = lifuren::config::CONFIG;
    if(input == wudaozi_input) {
        config.wudaozi = file;
    } else {
        SPDLOG_DEBUG("没有匹配元素");
    }
}

static void chooseDirectoryCallback(const wxCommandEvent&, wxTextCtrl* input) {
    auto path = lifuren::directory_chooser(wxT("选择目录"));
    if(path.empty()) {
        return;
    }
    input->Clear();
    input->AppendText(wxString::FromUTF8(path));
    auto& config = lifuren::config::CONFIG;
    if(input == tmp_input) {
        config.tmp = path;
    } else {
        SPDLOG_DEBUG("没有匹配元素");
    }
}
