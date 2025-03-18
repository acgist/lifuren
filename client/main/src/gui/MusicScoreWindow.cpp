#include "lifuren/GUI.hpp"

#include "wx/wx.h"
#include "wx/filesys.h"
#include "wx/mstream.h"
#include "wx/webview.h"
#include "wx/filename.h"
#include "wx/stdpaths.h"

#include "spdlog/spdlog.h"

#include "lifuren/Raii.hpp"

static wxPanel * panel     { nullptr };
static wxButton* open_score{ nullptr };
static wxButton* play_score{ nullptr };
static wxButton* rule_score{ nullptr };
static wxButton* plus_score{ nullptr };
static wxButton* fall_score{ nullptr };
static wxButton* pdf_score { nullptr };
static wxButton* img_score { nullptr };
#if wxUSE_WEBVIEW
static wxWebView* web_view{ nullptr };
#endif

static auto open_score_text = wxT("打开");
static auto play_score_text = wxT("钢琴演奏");
static auto rule_score_text = wxT("钢琴指法");
static auto plus_score_text = wxT("升调");
static auto fall_score_text = wxT("降调");
static auto pdf_score_text  = wxT("保存文档");
static auto img_score_text  = wxT("保存图片");
// 放大 缩小

static const auto open_score_id = 4000;
static const auto play_score_id = 4001;
static const auto rule_score_id = 4002;
static const auto plus_score_id = 4003;
static const auto fall_score_id = 4004;
static const auto pdf_score_id  = 4005;
static const auto img_score_id  = 4006;

lifuren::MusicScoreWindow::MusicScoreWindow(int width, int height, const wxString& title) : Window(width, height, title) {
}

lifuren::MusicScoreWindow::~MusicScoreWindow() {
}


wxFileName GetStaticContent(wxString file)
{
	// Construct Filename
	wxFileName fname(wxStandardPaths::Get().GetExecutablePath().BeforeLast('\\') + 
					 wxFileName::GetPathSeparator()+ 
					 wxString("static") + 
					 wxFileName::GetPathSeparator() + 
					 "html" + 
					 wxFileName::GetPathSeparator() + 
					 file + wxString(".html"));

	wxString s = fname.GetFullPath();

	// Return full path, which is corrected by wxWidgets
	return fname;
}

class MyWebHandler : public wxWebViewHandler
{
public:
	MyWebHandler(const wxString& protocol)
		: wxWebViewHandler(protocol)
	{
		m_fs = new wxFileSystem();
	}

	~MyWebHandler()
	{
		wxDELETE(m_fs);
	}
	
	virtual wxFSFile* GetFile (const wxString &uri)
	{
		wxString content = uri.substr(9, uri.length()).BeforeLast('/');
		wxFileName path = GetStaticContent(content);

		// It does not make any difference if this is used or not. It fails in both cases.
		wxString url = wxFileSystem::FileNameToURL(path);

		if ( wxFileSystem::HasHandlerForPath(url) )
		{
			return m_fs->OpenFile(url);
		}
		return NULL;
	}	

private:
	wxFileSystem* m_fs;
};

void lifuren::MusicScoreWindow::drawElement() {
    const int w = wxSystemSettings::GetMetric(wxSYS_SCREEN_X, nullptr);
    const int h = wxSystemSettings::GetMetric(wxSYS_SCREEN_Y, nullptr);
    panel      = new wxPanel (this);
    open_score = new wxButton(panel, open_score_id, open_score_text, wxPoint( 10, 10), wxSize(80, 30));
    play_score = new wxButton(panel, play_score_id, play_score_text, wxPoint(100, 10), wxSize(80, 30));
    rule_score = new wxButton(panel, rule_score_id, rule_score_text, wxPoint(190, 10), wxSize(80, 30));
    plus_score = new wxButton(panel, plus_score_id, plus_score_text, wxPoint(280, 10), wxSize(80, 30));
    fall_score = new wxButton(panel, fall_score_id, fall_score_text, wxPoint(370, 10), wxSize(80, 30));
    pdf_score  = new wxButton(panel, pdf_score_id,  pdf_score_text,  wxPoint(460, 10), wxSize(80, 30));
    img_score  = new wxButton(panel, img_score_id,  img_score_text,  wxPoint(550, 10), wxSize(80, 30));
    if(wxWebView::IsBackendAvailable(wxWebViewBackendEdge) || wxWebView::IsBackendAvailable(wxWebViewBackendWebKit)) {
        web_view = wxWebView::New(panel, wxID_ANY, wxWebViewDefaultURLStr, wxPoint(10, 50), wxSize(w - 20, h - 120));
        #if defined(_DEBUG) || !defined(NDEBUG)
        web_view->EnableAccessToDevTools(true);
        #else
        web_view->EnableAccessToDevTools(false);
        #endif
        auto x = GetStaticContent("index").GetFullPath();
		web_view->RegisterHandler(wxSharedPtr<wxWebViewHandler>(new MyWebHandler("static")));
		web_view->LoadURL(GetStaticContent("index").GetFullPath());
        // web_view->LoadURL("myscheme:index.html");
        // web_view->Bind(wxEVT_WEBVIEW_SCRIPT_MESSAGE_RECEIVED, [](const wxWebViewEvent& event) {
        //     wxLogMessage("Script message received; value = %s, handler = %s", event.GetString(), event.GetMessageHandler());
        // });
    }
}

void lifuren::MusicScoreWindow::bindEvent() {
}

void lifuren::MusicScoreWindow::fillData() {
    // auto ret = browser->load("web/index.html");
    // SPDLOG_DEBUG("浏览结果：{}", ret);
}
