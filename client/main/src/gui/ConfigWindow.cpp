#include "lifuren/GUI.hpp"

#include "wx/wx.h"

#include "spdlog/spdlog.h"

#include "lifuren/Config.hpp"

static wxPanel   * panel                 { nullptr };
static wxTextCtrl* tmp_input             { nullptr };
static wxButton  * tmp_button            { nullptr };
static wxTextCtrl* output_input          { nullptr };
static wxButton  * output_button         { nullptr };
static wxTextCtrl* model_bach_input      { nullptr };
static wxButton  * model_bach_button     { nullptr };
static wxTextCtrl* model_chopin_input    { nullptr };
static wxButton  * model_chopin_button   { nullptr };
static wxTextCtrl* model_mozart_input    { nullptr };
static wxButton  * model_mozart_button   { nullptr };
static wxTextCtrl* model_shikuang_input  { nullptr };
static wxButton  * model_shikuang_button { nullptr };
static wxTextCtrl* model_beethoven_input { nullptr };
static wxButton  * model_beethoven_button{ nullptr };

static const int tmp_button_id             = 3000;
static const int output_button_id          = 3001;
static const int model_bach_button_id      = 3002;
static const int model_chopin_button_id    = 3003;
static const int model_mozart_button_id    = 3004;
static const int model_shikuang_button_id  = 3005;
static const int model_beethoven_button_id = 3006;

const auto tmp_input_text              = wxT("临时目录");
const auto tmp_button_text             = wxT("选择临时目录");
const auto output_input_text           = wxT("输出目录");
const auto output_button_text          = wxT("选择输出目录");
const auto model_bach_input_text       = wxT("音频识谱模型文件");
const auto model_bach_button_text      = wxT("选择音频识谱模型文件");
const auto model_chopin_input_text     = wxT("简谱识谱模型文件");
const auto model_chopin_button_text    = wxT("选择简谱识谱模型文件");
const auto model_mozart_input_text     = wxT("五线谱识谱模型文件");
const auto model_mozart_button_text    = wxT("选择五线谱识谱模型文件");
const auto model_shikuang_input_text   = wxT("音频风格迁移模型文件");
const auto model_shikuang_button_text  = wxT("选择音频风格迁移模型文件");
const auto model_beethoven_input_text  = wxT("钢琴指法模型文件");
const auto model_beethoven_button_text = wxT("选择钢琴指法模型文件");

static void chooseFileCallback     (const wxCommandEvent&, wxTextCtrl*);
static void chooseDirectoryCallback(const wxCommandEvent&, wxTextCtrl*);

lifuren::ConfigWindow::ConfigWindow(int width, int height, const wxString& title) : Window(width, height, title) {
}

lifuren::ConfigWindow::~ConfigWindow() {
    lifuren::config::CONFIG.saveFile();
}

void lifuren::ConfigWindow::drawElement() {
    const int w = this->GetClientSize().GetWidth();
    const int h = this->GetClientSize().GetHeight();
    panel                  = new wxPanel(this);
    tmp_input              = new wxTextCtrl(panel, wxID_ANY,                  tmp_input_text             , wxPoint((w - 640) / 2,         10), wxSize(400, 30));
    tmp_button             = new wxButton  (panel, tmp_button_id,             tmp_button_text            , wxPoint((w - 640) / 2 + 410,   10), wxSize(240, 30));
    output_input           = new wxTextCtrl(panel, wxID_ANY,                  output_input_text          , wxPoint((w - 640) / 2,         50), wxSize(400, 30));
    output_button          = new wxButton  (panel, output_button_id,          output_button_text         , wxPoint((w - 640) / 2 + 410,   50), wxSize(240, 30));
    model_bach_input       = new wxTextCtrl(panel, wxID_ANY,                  model_bach_input_text      , wxPoint((w - 640) / 2,         90), wxSize(400, 30));
    model_bach_button      = new wxButton  (panel, model_bach_button_id,      model_bach_button_text     , wxPoint((w - 640) / 2 + 410,   90), wxSize(240, 30));
    model_chopin_input     = new wxTextCtrl(panel, wxID_ANY,                  model_chopin_input_text    , wxPoint((w - 640) / 2,        130), wxSize(400, 30));
    model_chopin_button    = new wxButton  (panel, model_chopin_button_id,    model_chopin_button_text   , wxPoint((w - 640) / 2 + 410,  130), wxSize(240, 30));
    model_mozart_input     = new wxTextCtrl(panel, wxID_ANY,                  model_mozart_input_text    , wxPoint((w - 640) / 2,        170), wxSize(400, 30));
    model_mozart_button    = new wxButton  (panel, model_mozart_button_id,    model_mozart_button_text   , wxPoint((w - 640) / 2 + 410,  170), wxSize(240, 30));
    model_shikuang_input   = new wxTextCtrl(panel, wxID_ANY,                  model_shikuang_input_text  , wxPoint((w - 640) / 2,        210), wxSize(400, 30));
    model_shikuang_button  = new wxButton  (panel, model_shikuang_button_id,  model_shikuang_button_text , wxPoint((w - 640) / 2 + 410,  210), wxSize(240, 30));
    model_beethoven_input  = new wxTextCtrl(panel, wxID_ANY,                  model_beethoven_input_text , wxPoint((w - 640) / 2,        250), wxSize(400, 30));
    model_beethoven_button = new wxButton  (panel, model_beethoven_button_id, model_beethoven_button_text, wxPoint((w - 640) / 2 + 410,  250), wxSize(240, 30));
}

void lifuren::ConfigWindow::bindEvent() {
    this->Bind(wxEVT_BUTTON, [](const wxCommandEvent& event) {
        const auto id = event.GetId();
        switch(id) {
            case tmp_button_id            : chooseDirectoryCallback(event, tmp_input);        break;
            case output_button_id         : chooseDirectoryCallback(event, output_input);     break;
            case model_bach_button_id     : chooseFileCallback(event, model_bach_input);      break;
            case model_chopin_button_id   : chooseFileCallback(event, model_chopin_input);    break;
            case model_mozart_button_id   : chooseFileCallback(event, model_mozart_input);    break;
            case model_shikuang_button_id : chooseFileCallback(event, model_shikuang_input);  break;
            case model_beethoven_button_id: chooseFileCallback(event, model_beethoven_input); break;
        }
    });
}

void lifuren::ConfigWindow::fillData() {
    const auto& config = lifuren::config::CONFIG;
    tmp_input            ->Clear();
    output_input         ->Clear();
    model_bach_input     ->Clear();
    model_chopin_input   ->Clear();
    model_mozart_input   ->Clear();
    model_shikuang_input ->Clear();
    model_beethoven_input->Clear();
    tmp_input            ->AppendText(wxString::FromUTF8(config.tmp.c_str()            ));
    output_input         ->AppendText(wxString::FromUTF8(config.output.c_str()         ));
    model_bach_input     ->AppendText(wxString::FromUTF8(config.model_bach.c_str()     ));
    model_chopin_input   ->AppendText(wxString::FromUTF8(config.model_chopin.c_str()   ));
    model_mozart_input   ->AppendText(wxString::FromUTF8(config.model_mozart.c_str()   ));
    model_shikuang_input ->AppendText(wxString::FromUTF8(config.model_shikuang.c_str() ));
    model_beethoven_input->AppendText(wxString::FromUTF8(config.model_beethoven.c_str()));
}

static void chooseFileCallback(const wxCommandEvent&, wxTextCtrl* input) {
    auto file = lifuren::file_chooser(wxT("选择模型"), wxT("模型文件|*.pt;*.pth"));
    if(file.empty()) {
        return;
    }
    input->Clear();
    input->AppendText(wxString::FromUTF8(file));
    auto& config = lifuren::config::CONFIG;
    if(input == model_bach_input) {
        config.model_bach = file;
    } else if(input == model_chopin_input) {
        config.model_chopin = file;
    } else if(input == model_mozart_input) {
        config.model_mozart = file;
    } else if(input == model_shikuang_input) {
        config.model_shikuang = file;
    } else if(input == model_beethoven_input) {
        config.model_beethoven = file;
    } else {
        SPDLOG_DEBUG("没有匹配元素");
    }
}

static void chooseDirectoryCallback(const wxCommandEvent&, wxTextCtrl* input) {
    auto file = lifuren::directory_chooser(wxT("选择目录"));
    if(file.empty()) {
        return;
    }
    auto& config = lifuren::config::CONFIG;
    input->Clear();
    input->AppendText(wxString::FromUTF8(file));
    if(input == tmp_input) {
        config.tmp = file;
    } else if(input == output_input) {
        config.output = file;
    } else {
        SPDLOG_DEBUG("没有匹配元素");
    }
}
