#include "lifuren/FLTK.hpp"

#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"

static Fl_Choice* clientPtr{ nullptr };
static Fl_Input*  apiPtr   { nullptr };
static Fl_Input*  usernamePtr{ nullptr };
static Fl_Input*  passwordPtr{ nullptr };
static Fl_Choice* authTypePtr{ nullptr };
static Fl_Input*  chatPathPtr { nullptr };
static Fl_Input*  chatModelPtr{ nullptr };
static Fl_Input*  embeddingPathPtr { nullptr };
static Fl_Input*  embeddingModelPtr{ nullptr };

lifuren::ChatConfigWindow::ChatConfigWindow(int width, int height, const char* title) : ConfigWindow(width, height, title) {
    this->chatConfigPtr = &lifuren::config::CONFIG.chat;
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
    LFR_DELETE_PTR(embeddingPathPtr);
    LFR_DELETE_PTR(embeddingModelPtr);
}

void lifuren::ChatConfigWindow::saveConfig() {
    if(this->chatConfigPtr->client == "ollama") {
        lifuren::config::OllamaConfig& config = lifuren::config::CONFIG.ollama;
        config.api      = apiPtr->value();
        config.username = usernamePtr->value();
        config.password = passwordPtr->value();
        // config.authType = authTypePtr->text();
        auto& chatClient = config.chatClient;
        chatClient.path  = chatPathPtr->value();
        chatClient.model = chatModelPtr->value();
        auto& embeddingClient = config.embeddingClient;
        embeddingClient.path  = embeddingPathPtr->value();
        embeddingClient.model = embeddingModelPtr->value();
    } else {
    }
    lifuren::ConfigWindow::saveConfig();
}

void lifuren::ChatConfigWindow::redrawConfigElement() {
    if(this->chatConfigPtr->client == "ollama") {
        lifuren::config::OllamaConfig& config = lifuren::config::CONFIG.ollama;
        apiPtr->value(config.api.c_str());
        usernamePtr->value(config.username.c_str());
        passwordPtr->value(config.password.c_str());
        auto defaultPtr = authTypePtr->find_item(config.authType.c_str());
        if(defaultPtr) {
            // defaultPtr = authTypePtr->find_item("NONE");
            authTypePtr->value(defaultPtr); 
        }
        auto& chatClient = config.chatClient;
        chatPathPtr->value(chatClient.path.c_str());
        chatModelPtr->value(chatClient.model.c_str());
        auto& embeddingClient = config.embeddingClient;
        embeddingPathPtr->value(embeddingClient.path.c_str());
        embeddingModelPtr->value(embeddingClient.model.c_str());
    } else {
    }
}

// TODO: 优化模型通过接口查询回来
void lifuren::ChatConfigWindow::drawElement() {
    clientPtr    = new Fl_Choice(110, 10,  this->w() - 200, 30, "终端名称");
    apiPtr       = new Fl_Input(110,  50,  this->w() - 200, 30, "服务地址");
    usernamePtr  = new Fl_Input(110,  90,  this->w() - 200, 30, "账号");
    passwordPtr  = new Fl_Input(110,  130, this->w() - 200, 30, "密码");
    authTypePtr  = new Fl_Choice(110, 170, this->w() - 200, 30, "授权类型"); 
    chatPathPtr  = new Fl_Input(110,  210, this->w() - 200, 30, "LLM地址");
    chatModelPtr = new Fl_Input(110,  250, this->w() - 200, 30, "LLM模型");
    embeddingPathPtr  = new Fl_Input(110, 290, this->w() - 200, 30, "Embedding地址");
    embeddingModelPtr = new Fl_Input(110, 330, this->w() - 200, 30, "Embedding模型");
    // 终端名称选择
    std::for_each(lifuren::config::chatClients.begin(), lifuren::config::chatClients.end(), [](const auto& v) {
        clientPtr->add(v.c_str());
    });
    auto defaultPtr = clientPtr->find_item(this->chatConfigPtr->client.c_str());
    if(defaultPtr) {
        clientPtr->value(defaultPtr); 
    }
    clientPtr->callback([](Fl_Widget*, void* voidPtr) {
        ChatConfigWindow* windowPtr = static_cast<ChatConfigWindow*>(voidPtr);
        windowPtr->chatConfigPtr->client = clientPtr->text();
        windowPtr->redrawConfigElement();
    }, this);
    // 授权类型选择
    authTypePtr->add("NONE");
    authTypePtr->add("Basic");
    authTypePtr->add("Token");
    authTypePtr->callback([](Fl_Widget*, void*) {
        lifuren::config::OllamaConfig& config = lifuren::config::CONFIG.ollama;
        config.authType = authTypePtr->text();
    }, this);
    this->redrawConfigElement();
}
