#include "lifuren/GUI.hpp"

#include <cmath>
#include <algorithm>

#include "wx/wx.h"
#include <wx/filename.h>
#include <wx/stdpaths.h>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/Config.hpp"

static std::string last_directory = "";

bool lifuren::Application::OnInit() {
    static lifuren::MainWindow* main = new lifuren::MainWindow(LFR_WINDOW_WIDTH, LFR_WINDOW_HEIGHT);
    main->init();
    main->Show(true);
    return true;
}

void lifuren::initGUI() {
    SPDLOG_INFO("启动GUI");
    static Application* app = new Application();
    app->SetAppName(wxT("李夫人"));
    app->SetVendorName(wxT("acgist"));
    wxApp::SetInstance(app);
    #if _WIN32
    wxEntry();
    #else
    int    argc = 0;
    char** argv = nullptr;
    wxEntry(argc, argv);
    #endif
    SPDLOG_INFO("结束GUI");
}

std::string lifuren::file_chooser(const wxString& title, const wxString& filter, const wxString& directory) {
    wxString last_directory = directory;
    if(last_directory.empty()) {
        last_directory = wxString::FromUTF8(::last_directory);
    }
    wxFileDialog chooser(nullptr, title, last_directory, "", filter, wxFD_OPEN);
    if(chooser.ShowModal() == wxID_CANCEL) {
        SPDLOG_DEBUG("取消选择文件：{}", title.mb_str(wxConvUTF8).data());
        return {};
    }
    std::string filename = chooser.GetPath().mb_str(wxConvUTF8).data();
    last_directory = lifuren::file::parent(filename);
    SPDLOG_DEBUG("选择文件：{}", filename);
    return filename;
}

std::string lifuren::directory_chooser(const wxString& title, const wxString& directory) {
    wxString last_directory = directory;
    if(last_directory.empty()) {
        last_directory = wxString::FromUTF8(::last_directory);
    }
    wxDirDialog chooser(nullptr, title, last_directory);
    if(chooser.ShowModal() == wxID_CANCEL) {
        SPDLOG_DEBUG("取消选择目录：{}", title.mb_str(wxConvUTF8).data());
        return {};
    }
    std::string filename = chooser.GetPath().mb_str(wxConvUTF8).data();
    last_directory = filename;
    SPDLOG_DEBUG("选择目录：{}", filename);
    return filename;
}

wxString lifuren::app_base_dir(const wxString& path) {
    return wxFileName(wxStandardPaths::Get().GetExecutablePath().BeforeLast(wxFileName::GetPathSeparator()) + wxFileName::GetPathSeparator() + path).GetFullPath();
}

lifuren::Window::Window(int width, int height, const wxString& title) : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(width, height)) {
    this->SetTitle(title);
    SPDLOG_DEBUG("打开窗口：{}", this->GetTitle().mb_str(wxConvUTF8).data());
}

lifuren::Window::~Window() {
    SPDLOG_DEBUG("关闭窗口：{}", this->GetTitle().mb_str(wxConvUTF8).data());
}

void lifuren::Window::init() {
    this->Centre();
    this->loadIcon();
    this->drawWidget();
    this->bindEvent();
    this->fillData();
}

void lifuren::Window::loadIcon() {
    this->SetIcon(wxIcon(app_base_dir("logo.ico"), wxBITMAP_TYPE_ICO));
}

void lifuren::Window::drawWidget() {
}

void lifuren::Window::bindEvent() {
}

void lifuren::Window::fillData() {
}
