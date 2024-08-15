#include "lifuren/FLTK.hpp"

#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"

static Fl_Choice* clientPtr{ nullptr };
static Fl_Input*  apiPtr   { nullptr };
static Fl_Input*  usernamePtr { nullptr };
static Fl_Input*  passwordPtr { nullptr };
static Fl_Choice* authTypePtr { nullptr };
static Fl_Input*  chatPathPtr { nullptr };
static Fl_Input*  chatModelPtr{ nullptr };

static void clientCallback(Fl_Widget*, void*);

lifuren::ChatConfigWindow::ChatConfigWindow(int width, int height, const char* title) : ConfigWindow(width, height, title) {
}

lifuren::ChatConfigWindow::~ChatConfigWindow() {
    this->saveConfig();
    LFR_DELETE_PTR(clientPtr);
    LFR_DELETE_PTR(apiPtr);
    LFR_DELETE_PTR(usernamePtr);
    LFR_DELETE_PTR(passwordPtr);
    LFR_DELETE_PTR(authTypePtr);
    LFR_DELETE_PTR(chatPathPtr);
    LFR_DELETE_PTR(chatModelPtr);
}

void lifuren::ChatConfigWindow::saveConfig() {
    lifuren::config::ChatConfig& chatConfig = lifuren::config::CONFIG.chat;
    if(chatConfig.client == "ollama") {
        auto& ollamaConfig    = lifuren::config::CONFIG.ollama;
        ollamaConfig.api      = apiPtr->value();
        ollamaConfig.username = usernamePtr->value();
        ollamaConfig.password = passwordPtr->value();
        LFR_CHOICE_GET_DEFAULT(ollamaConfig.authType, authTypePtr);
        auto& chatClientConfig = ollamaConfig.chatClient;
        chatClientConfig.path  = chatPathPtr->value();
        chatClientConfig.model = chatModelPtr->value();
    } else {
    }
    lifuren::ConfigWindow::saveConfig();
}

void lifuren::ChatConfigWindow::redrawConfigElement() {
    lifuren::config::ChatConfig& chatConfig = lifuren::config::CONFIG.chat;
    if(chatConfig.client == "ollama") {
        const auto& ollamaConfig = lifuren::config::CONFIG.ollama;
        apiPtr->value(ollamaConfig.api.c_str());
        usernamePtr->value(ollamaConfig.username.c_str());
        passwordPtr->value(ollamaConfig.password.c_str());
        LFR_CHOICE_SET_DEFAULT(authTypePtr, ollamaConfig.authType);
        const auto& chatClientConfig = ollamaConfig.chatClient;
        chatPathPtr->value(chatClientConfig.path.c_str());
        chatModelPtr->value(chatClientConfig.model.c_str());
    } else {
    }
}

// TODO: 优化模型通过接口查询回来
void lifuren::ChatConfigWindow::drawElement() {
    // 布局
    clientPtr    = new Fl_Choice(110, 10,  200,             30, "终端名称");
    apiPtr       = new Fl_Input(110,  50,  this->w() - 200, 30, "服务地址");
    usernamePtr  = new Fl_Input(110,  90,  this->w() - 200, 30, "账号");
    passwordPtr  = new Fl_Input(110,  130, this->w() - 200, 30, "密码");
    authTypePtr  = new Fl_Choice(110, 170, 200,             30, "授权类型"); 
    chatPathPtr  = new Fl_Input(110,  210, this->w() - 200, 30, "请求地址");
    chatModelPtr = new Fl_Input(110,  250, this->w() - 200, 30, "语言模型");
    // 事件
    // 终端名称
    const auto& chatConfig = lifuren::config::CONFIG.chat;
    std::for_each(chatConfig.clients.begin(), chatConfig.clients.end(), [](const auto& v) {
        lifuren::config::ChatConfig& chatConfig = lifuren::config::CONFIG.chat;
        int value = clientPtr->add(v.c_str());
        if(v == chatConfig.client) {
            clientPtr->value(value);
        }
    });
    clientPtr->callback(clientCallback, this);
    // 授权类型
    authTypePtr->add("NONE");
    authTypePtr->add("Basic");
    authTypePtr->add("Token");
    // 重绘配置
    this->redrawConfigElement();
}

static void clientCallback(Fl_Widget*, void* voidPtr) {
    lifuren::ChatConfigWindow* windowPtr = static_cast<lifuren::ChatConfigWindow*>(voidPtr);
    lifuren::config::ChatConfig& chatConfig = lifuren::config::CONFIG.chat;
    chatConfig.client = clientPtr->text();
    windowPtr->redrawConfigElement();
}
