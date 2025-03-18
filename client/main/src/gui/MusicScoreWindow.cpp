#include "lifuren/GUI.hpp"

#include "wx/wx.h"
#include "wx/filesys.h"
#include "wx/mstream.h"
#include "wx/webview.h"

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

class WxHtmlFSHandler : public wxWebViewHandler
{
    public:
    WxHtmlFSHandler( const wxString& scheme ) : wxWebViewHandler( scheme ) {
        std::cout << "++\n";
    }
    ~WxHtmlFSHandler() {
        std::cout << "----\n";
    }

    wxFSFile* GetFile( const wxString& uri ) override
    {
        std::cout << "=====\n";
        std::string html = "<!DOCTYPE html><html><head><meta http-equiv='content-type' content='text/html;"
        "charset=UTF-8'>\n<meta name='viewport' content='width=device-width,initial-scale=1.0'>"
        "</head><body>"
        "<h1>This is a test</h1>"
        "<a href=\"logo?2\"><img width=\"50%\" src=\"../pic1.png\"></a>"
        "<a href=\"logo?4\"><img width=\"50%\" src=\"pic2.png\"></a></body></html>";
        auto x = new wxMemoryInputStream(html.data(), html.size() );
        return new wxFSFile(x, uri, wxT( "text/html" ), ""
        #if wxUSE_DATETIME
        , wxDateTime::Now()
      #endif
    );
    };
};

static WxHtmlFSHandler* hx;

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
        hx = new WxHtmlFSHandler( "myscheme" );
        static auto xx = wxSharedPtr< wxWebViewHandler >( hx ) ;
        web_view->RegisterHandler(xx);
        web_view->SetPage(LR"(<!DOCTYPE HTML>
<html>

<head>
  <title>李夫人</title>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width" />
  <meta name="keywords" content="李夫人" />
  <meta name="description" content="李夫人" />

  <!-- <script type="text/javascript" src="./jspdf.es.min.js"></script> -->
  <script type="text/javascript" src="myscheme:opensheetmusicdisplay.min.js"></script>
</head>

<body>
  <div id="container">1</div>
  <img width="50%" src="myscheme:pic1.png">
  <script>
    const display = new opensheetmusicdisplay.OpenSheetMusicDisplay("container");
    display.setOptions({
      backend: "canvas",
      drawTitle: true,
    });
    display
      .load("myscheme:/music.xml")
      .then(() => {
        display.render();
      }
      );
  </script>
</body>

</html>)", {});
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
