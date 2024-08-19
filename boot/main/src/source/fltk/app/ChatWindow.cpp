#include "lifuren/FLTK.hpp"

#include <list>
#include <tuple>
#include <thread>
#include <algorithm>

#include "spdlog/spdlog.h"

#include "FL/fl_ask.H"
#include "Fl/Fl_Pack.H"
#include "FL/Fl_Button.H"
#include "Fl/Fl_Scroll.H"
#include "Fl/Fl_Choice.H"
#include "FL/Fl_Text_Buffer.H"
#include "FL/Fl_Text_Editor.H"
#include "FL/Fl_Text_Display.H"

#include "lifuren/RAG.hpp"
#include "lifuren/Strings.hpp"

#ifndef LFR_CHAT_STREAM
#define LFR_CHAT_STREAM true
#endif

static Fl_Button* sendPtr    { nullptr };
static Fl_Button* stopPtr    { nullptr };
static Fl_Button* configPtr  { nullptr };
static Fl_Choice* documentPtr{ nullptr };
static Fl_Scroll* scrollPtr  { nullptr };
static Fl_Pack*   packPtr    { nullptr };
static Fl_Text_Buffer* messageBufferPtr{ nullptr };
static Fl_Text_Editor* messageEditorPtr{ nullptr };

// 是否停止
static bool messageStop{ true  };
static bool messageWait{ false };
// 消息列表
static std::list<std::tuple<Fl_Text_Buffer*, Fl_Text_Display*, bool>> messageList{};
// 配置窗口
static lifuren::ChatConfigWindow* chatConfigWindowPtr{ nullptr };

static void documentCallback(Fl_Widget*, void*);
static void chatMessageThread(const lifuren::ChatWindow& window);
static void chatMessage(const char* message, const lifuren::ChatWindow& window);
static void appendMessage(const std::string& message, bool done);
static void waitForMessage();
static void doneForMessage();

lifuren::ChatWindow::ChatWindow(int width, int height, const char* title) : ModelWindow(width, height, title) {
    lifuren::options::RestChatOptions options;
    const auto& config = lifuren::config::CONFIG;
    const auto& chat   = config.chat;
    if(chat.client == lifuren::config::CONFIG_OLLAMA) {
        options.of(config.ollama);
        this->clientPtr = new OllamaChatClient{options};
    } else {
    }
}

lifuren::ChatWindow::~ChatWindow() {
    SPDLOG_DEBUG("关闭窗口：{}", __FILE__);
    messageStop = true;
    messageWait = false;
    std::for_each(messageList.begin(), messageList.end(), [](const auto& tuple) {
        delete std::get<1>(tuple);
        delete std::get<0>(tuple);
    });
    messageList.clear();
    LFR_DELETE_PTR(sendPtr);
    LFR_DELETE_PTR(stopPtr);
    LFR_DELETE_PTR(configPtr);
    LFR_DELETE_PTR(documentPtr);
    LFR_DELETE_PTR(packPtr);
    LFR_DELETE_PTR(scrollPtr);
    LFR_DELETE_PTR(messageEditorPtr);
    LFR_DELETE_PTR(messageBufferPtr);
    LFR_DELETE_PTR(chatConfigWindowPtr);
    LFR_DELETE_THIS_PTR(clientPtr);
}

void lifuren::ChatWindow::drawElement() {
    sendPtr     = new Fl_Button(this->w() - 120, this->h() - 50, 100, 30, "发送消息");
    stopPtr     = new Fl_Button(this->w() - 230, this->h() - 50, 100, 30, "结束回答");
    configPtr   = new Fl_Button(this->w() - 340, this->h() - 50, 100, 30, "⚙配置");
    documentPtr = new Fl_Choice(110,             this->h() - 50, 200, 30, "文档目录");
    scrollPtr   = new Fl_Scroll(10, 10, this->w() - 20, this->h() - 180);
    packPtr     = new Fl_Pack  (10, 10, this->w() - 40, this->h() - 180);
    scrollPtr->type(Fl_Scroll::VERTICAL_ALWAYS);
    packPtr->type(Fl_Pack::VERTICAL);
    packPtr->spacing(2);
    packPtr->end();
    scrollPtr->end();
    messageEditorPtr = new Fl_Text_Editor(10, this->h() - 160, this->w() - 20, 100);
    messageBufferPtr = new Fl_Text_Buffer();
    messageEditorPtr->buffer(messageBufferPtr);
    messageEditorPtr->wrap_mode(messageEditorPtr->WRAP_AT_COLUMN, messageEditorPtr->textfont());
    messageEditorPtr->end();
    // 事件
    // 文档目录
    const auto& documentMark = lifuren::config::CONFIG.documentMark;
    for(auto& value : documentMark) {
        std::string path = value.path;
        LFR_CHOICE_TRANSFER(path);
        documentPtr->add(path.c_str());
    }
    documentPtr->callback(documentCallback, this);
    // 回车发送
    messageEditorPtr->when(Fl_Event::FL_ENTER);
    messageEditorPtr->callback([](Fl_Widget*, void* voidPtr) {
        if(FL_Enter == Fl::event_key()) {
            ChatWindow* windowPtr = static_cast<ChatWindow*>(voidPtr);
            chatMessageThread(*windowPtr);
        }
    }, this);
    // 发送
    sendPtr->callback([](Fl_Widget*, void* voidPtr) {
        ChatWindow* windowPtr = static_cast<ChatWindow*>(voidPtr);
        chatMessageThread(*windowPtr);
    }, this);
    // 结束
    stopPtr->callback([](Fl_Widget*, void*) {
        messageStop = true;
    }, this);
    // 配置
    configPtr->callback([](Fl_Widget*, void* voidPtr) {
        chatConfigWindowPtr = new lifuren::ChatConfigWindow{LFR_WINDOW_WIDTH_CONFIG, LFR_WINDOW_HEIGHT_CONFIG};
        chatConfigWindowPtr->init();
        chatConfigWindowPtr->show();
        chatConfigWindowPtr->callback([](Fl_Widget*, void*) {
            chatConfigWindowPtr->hide();
            LFR_DELETE_PTR(chatConfigWindowPtr);
        }, voidPtr);
    }, this);
}

static void documentCallback(Fl_Widget*, void* voidPtr) {
    lifuren::ChatWindow* windowPtr = static_cast<lifuren::ChatWindow*>(voidPtr);
    const std::string path = documentPtr->text();
    if(path.empty()) {
        windowPtr->clientPtr->ragSearchEngine = nullptr;
        return;
    }
    auto& documentMark = lifuren::config::CONFIG.documentMark;
    auto iterator = std::find(documentMark.begin(), documentMark.end(), path);
    if(iterator != documentMark.end()) {
        auto ragClient = lifuren::RAGClient::getRAGClient(iterator->rag, iterator->path, iterator->embedding);
        ragClient->loadIndex();
        windowPtr->clientPtr->ragSearchEngine = std::move(ragClient);
        // windowPtr->clientPtr->ragSearchEngine.reset(ragClient.release());
    } else {
        SPDLOG_WARN("不支持的文档路径：{}", path);
    }
}

static void chatMessageThread(const lifuren::ChatWindow& window) {
    if(messageWait) {
        fl_message("等待上次消息响应完成");
        return;
    }
    char* message = messageBufferPtr->text();
    message = lifuren::strings::trim(message);
    if(std::strlen(message) <= 0) {
        fl_message("请输入发送的内容");
        return;
    }
    appendMessage(message, true);
    messageBufferPtr->text("");
    waitForMessage();
    std::thread thread{chatMessage, message, std::ref(window)};
    thread.detach();
}

static void chatMessage(const char* message, const lifuren::ChatWindow& window) {
    if(window.clientPtr == nullptr) {
        fl_message("没有聊天终端");
        return;
    }
    #if LFR_CHAT_STREAM
    window.clientPtr->chat(message, [](const char* text, size_t length, bool done) {
        appendMessage(std::string(text, length), done);
        if(done) {
            doneForMessage();
            return true;
        } else {
            return !messageStop;
        }
    });
    #else
    auto response = window.clientPtr->chat(message);
    appendMessage(response, done);
    doneForMessage();
    #endif
}

static void appendMessage(const std::string& message, bool done) {
    // TODO: 不要滚动
    const static int height = 100;
    bool append = true;
    if(messageList.empty()) {
        // 新的消息
    } else {
        auto& oldMessage = messageList.back();
        auto& oldDone    = std::get<2>(oldMessage);
        if(oldDone) {
            // 新的消息
        } else {
            auto& oldBufferPtr = std::get<0>(oldMessage);
            oldBufferPtr->append(message.c_str());
            oldDone = done;
            return;
        }
    }
    scrollPtr->begin();
    packPtr->begin();
    Fl_Text_Display* displayPtr = new Fl_Text_Display(0, 0, scrollPtr->w() - 40, height);
    Fl_Text_Buffer*  bufferPtr  = new Fl_Text_Buffer();
    displayPtr->buffer(bufferPtr);
    displayPtr->wrap_mode(displayPtr->WRAP_AT_COLUMN, displayPtr->textfont());
    bufferPtr->text(message.c_str());
    displayPtr->end();
    packPtr->redraw();
    packPtr->end();
    scrollPtr->redraw();
    scrollPtr->end();
    scrollPtr->scroll_to(0, packPtr->h() - scrollPtr->h() + height);
    messageList.push_back(std::make_tuple(bufferPtr, displayPtr, done));
}

static void waitForMessage() {
    messageWait = true;
    messageStop = false;
    messageEditorPtr->color(FL_GRAY);
    messageEditorPtr->redraw();
}

static void doneForMessage() {
    messageWait = false;
    messageStop = true;
    messageEditorPtr->color(FL_WHITE);
    messageEditorPtr->redraw();
}
