#include "lifuren/GUI.hpp"

#include "wx/wx.h"

#include "spdlog/spdlog.h"

#include "lifuren/Config.hpp"

static wxPanel   * panel               { nullptr };
static wxTextCtrl* tmp_input           { nullptr };
static wxButton  * tmp_button          { nullptr };
static wxTextCtrl* output_input        { nullptr };
static wxButton  * output_button       { nullptr };
static wxTextCtrl* model_wudaozi_input { nullptr };
static wxButton  * model_wudaozi_button{ nullptr };

static const int tmp_button_id           = 3000;
static const int output_button_id        = 3001;
static const int model_wudaozi_button_id = 3002;

static const auto tmp_input_text            = wxT("临时目录");
static const auto tmp_button_text           = wxT("选择临时目录");
static const auto output_input_text         = wxT("输出目录");
static const auto output_button_text        = wxT("选择输出目录");
static const auto model_wudaozi_input_text  = wxT("视频生成模型文件");
static const auto model_wudaozi_button_text = wxT("选择视频生成模型文件");

static void chooseFileCallback     (const wxCommandEvent&, wxTextCtrl*);
static void chooseDirectoryCallback(const wxCommandEvent&, wxTextCtrl*);

lifuren::ConfigWindow::ConfigWindow(int width, int height, const wxString& title) : Window(width, height, title) {
}

lifuren::ConfigWindow::~ConfigWindow() {
    lifuren::config::CONFIG.saveFile();
    panel                = nullptr;
    tmp_input            = nullptr;
    tmp_button           = nullptr;
    output_input         = nullptr;
    output_button        = nullptr;
    model_wudaozi_input  = nullptr;
    model_wudaozi_button = nullptr;
}

void lifuren::ConfigWindow::drawWidget() {
    const int w = this->GetClientSize().GetWidth();
    const int h = this->GetClientSize().GetHeight();
    panel                = new wxPanel(this);
    tmp_input            = new wxTextCtrl(panel, wxID_ANY,                tmp_input_text,            wxPoint((w - 640) / 2,        10), wxSize(400, 30));
    tmp_button           = new wxButton  (panel, tmp_button_id,           tmp_button_text,           wxPoint((w - 640) / 2 + 410,  10), wxSize(240, 30));
    output_input         = new wxTextCtrl(panel, wxID_ANY,                output_input_text,         wxPoint((w - 640) / 2,        50), wxSize(400, 30));
    output_button        = new wxButton  (panel, output_button_id,        output_button_text,        wxPoint((w - 640) / 2 + 410,  50), wxSize(240, 30));
    model_wudaozi_input  = new wxTextCtrl(panel, wxID_ANY,                model_wudaozi_input_text,  wxPoint((w - 640) / 2,        90), wxSize(400, 30));
    model_wudaozi_button = new wxButton  (panel, model_wudaozi_button_id, model_wudaozi_button_text, wxPoint((w - 640) / 2 + 410,  90), wxSize(240, 30));
}

void lifuren::ConfigWindow::bindEvent() {
    this->Bind(wxEVT_BUTTON, [](const wxCommandEvent& event) {
        const auto id = event.GetId();
        switch(id) {
            case tmp_button_id          : chooseDirectoryCallback(event, tmp_input);      break;
            case output_button_id       : chooseDirectoryCallback(event, output_input);   break;
            case model_wudaozi_button_id: chooseFileCallback(event, model_wudaozi_input); break;
        }
    });
}

void lifuren::ConfigWindow::fillData() {
    const auto& config = lifuren::config::CONFIG;
    tmp_input          ->Clear();
    output_input       ->Clear();
    model_wudaozi_input->Clear();
    tmp_input          ->AppendText(wxString::FromUTF8(config.tmp.c_str()           ));
    output_input       ->AppendText(wxString::FromUTF8(config.output.c_str()        ));
    model_wudaozi_input->AppendText(wxString::FromUTF8(config.model_wudaozi.c_str() ));
}

static void chooseFileCallback(const wxCommandEvent&, wxTextCtrl* input) {
    auto file = lifuren::file_chooser(wxT("选择模型"), wxT("模型文件|*.pt;*.pth"));
    if(file.empty()) {
        return;
    }
    input->Clear();
    input->AppendText(wxString::FromUTF8(file));
    auto& config = lifuren::config::CONFIG;
    if(input == model_wudaozi_input) {
        config.model_wudaozi = file;
    } else {
        SPDLOG_DEBUG("没有匹配元素");
    }
}

static void chooseDirectoryCallback(const wxCommandEvent&, wxTextCtrl* input) {
    auto file = lifuren::directory_chooser(wxT("选择目录"));
    if(file.empty()) {
        return;
    }
    input->Clear();
    input->AppendText(wxString::FromUTF8(file));
    auto& config = lifuren::config::CONFIG;
    if(input == tmp_input) {
        config.tmp = file;
    } else if(input == output_input) {
        config.output = file;
    } else {
        SPDLOG_DEBUG("没有匹配元素");
    }
}
