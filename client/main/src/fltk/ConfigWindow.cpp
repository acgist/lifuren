#include "lifuren/FLTK.hpp"

#include "spdlog/spdlog.h"

#include "FL/Fl_Input.H"
#include "FL/Fl_Button.H"
#include "FL/Fl_Choice.H"

#include "lifuren/Raii.hpp"
#include "lifuren/Config.hpp"

static Fl_Input * tmpPtr            { nullptr };
static Fl_Button* tmpBPtr           { nullptr };
static Fl_Input * outputPtr         { nullptr };
static Fl_Button* outputBPtr        { nullptr };
static Fl_Input * modelBachPtr      { nullptr };
static Fl_Button* modelBachBPtr     { nullptr };
static Fl_Input * modelChopinPtr    { nullptr };
static Fl_Button* modelChopinBPtr   { nullptr };
static Fl_Input * modelMozartPtr    { nullptr };
static Fl_Button* modelMozartBPtr   { nullptr };
static Fl_Input * modelWudaoziPtr   { nullptr };
static Fl_Button* modelWudaoziBPtr  { nullptr };
static Fl_Input * modelShikuangPtr  { nullptr };
static Fl_Button* modelShikuangBPtr { nullptr };
static Fl_Input * modelBeethovenPtr { nullptr };
static Fl_Button* modelBeethovenBPtr{ nullptr };

static void chooseFileCallback     (Fl_Widget*, void*);
static void chooseDirectoryCallback(Fl_Widget*, void*);

lifuren::ConfigWindow::ConfigWindow(int width, int height, const char* title) : Window(width, height, title) {
}

lifuren::ConfigWindow::~ConfigWindow() {
    LFR_DELETE_PTR(tmpPtr            );
    LFR_DELETE_PTR(tmpBPtr           );
    LFR_DELETE_PTR(outputPtr         );
    LFR_DELETE_PTR(outputBPtr        );
    LFR_DELETE_PTR(modelBachPtr      );
    LFR_DELETE_PTR(modelBachBPtr     );
    LFR_DELETE_PTR(modelChopinPtr    );
    LFR_DELETE_PTR(modelChopinBPtr   );
    LFR_DELETE_PTR(modelMozartPtr    );
    LFR_DELETE_PTR(modelMozartBPtr   );
    LFR_DELETE_PTR(modelWudaoziPtr   );
    LFR_DELETE_PTR(modelWudaoziBPtr  );
    LFR_DELETE_PTR(modelShikuangPtr  );
    LFR_DELETE_PTR(modelShikuangBPtr );
    LFR_DELETE_PTR(modelBeethovenPtr );
    LFR_DELETE_PTR(modelBeethovenBPtr);
    lifuren::config::CONFIG.saveFile();
}

void lifuren::ConfigWindow::drawElement() {
    tmpPtr             = new Fl_Input (160,   20, 400, 30, "临时目录");
    tmpBPtr            = new Fl_Button(560,   20, 180, 30, "选择临时目录");
    outputPtr          = new Fl_Input (160,   60, 400, 30, "输出目录");
    outputBPtr         = new Fl_Button(560,   60, 180, 30, "选择输出目录");
    modelBeethovenPtr  = new Fl_Input (160,  100, 400, 30, "钢琴指法模型文件");
    modelBeethovenBPtr = new Fl_Button(560,  100, 180, 30, "选择钢琴指法模型文件");
    modelBachPtr       = new Fl_Input (160,  140, 400, 30, "音频识谱模型文件");
    modelBachBPtr      = new Fl_Button(560,  140, 180, 30, "选择音频识谱模型文件");
    modelChopinPtr     = new Fl_Input (160,  180, 400, 30, "简谱识谱模型文件");
    modelChopinBPtr    = new Fl_Button(560,  180, 180, 30, "选择简谱识谱模型文件");
    modelMozartPtr     = new Fl_Input (160,  220, 400, 30, "五线谱识谱模型文件");
    modelMozartBPtr    = new Fl_Button(560,  220, 180, 30, "选择五线谱识谱模型文件");
    modelShikuangPtr   = new Fl_Input (160,  260, 400, 30, "音频风格迁移模型文件");
    modelShikuangBPtr  = new Fl_Button(560,  260, 180, 30, "选择音频风格迁移模型文件");
    modelWudaoziPtr    = new Fl_Input (160,  300, 400, 30, "图片风格迁移模型文件");
    modelWudaoziBPtr   = new Fl_Button(560,  300, 180, 30, "选择图片风格迁移模型文件");
}

void lifuren::ConfigWindow::bindEvent() {
    tmpBPtr           ->callback(chooseDirectoryCallback, tmpPtr           );
    outputBPtr        ->callback(chooseDirectoryCallback, outputPtr        );
    modelBachBPtr     ->callback(chooseFileCallback,      modelBachPtr     );
    modelChopinBPtr   ->callback(chooseFileCallback,      modelChopinPtr   );
    modelMozartBPtr   ->callback(chooseFileCallback,      modelMozartPtr   );
    modelWudaoziBPtr  ->callback(chooseFileCallback,      modelWudaoziPtr  );
    modelShikuangBPtr ->callback(chooseFileCallback,      modelShikuangPtr );
    modelBeethovenBPtr->callback(chooseFileCallback,      modelBeethovenPtr);
}

void lifuren::ConfigWindow::fillData() {
    const auto& config = lifuren::config::CONFIG;
    tmpPtr           ->value(config.tmp.c_str()            );
    outputPtr        ->value(config.output.c_str()         );
    modelBachPtr     ->value(config.model_bach.c_str()     );
    modelChopinPtr   ->value(config.model_chopin.c_str()   );
    modelMozartPtr   ->value(config.model_mozart.c_str()   );
    modelWudaoziPtr  ->value(config.model_wudaozi.c_str()  );
    modelShikuangPtr ->value(config.model_shikuang.c_str() );
    modelBeethovenPtr->value(config.model_beethoven.c_str());
}

static void chooseFileCallback(Fl_Widget* widget, void* voidPtr) {
    lifuren::fileChooser(widget, voidPtr, "选择文件", "*.{pt,pth}");
    auto& config = lifuren::config::CONFIG;
    if(voidPtr == modelBachPtr) {
        config.model_bach = modelBachPtr->value();
    } else if(voidPtr == modelChopinPtr) {
        config.model_chopin = modelChopinPtr->value();
    } else if(voidPtr == modelMozartPtr) {
        config.model_mozart = modelMozartPtr->value();
    } else if(voidPtr == modelWudaoziPtr) {
        config.model_wudaozi = modelWudaoziPtr->value();
    } else if(voidPtr == modelShikuangPtr) {
        config.model_shikuang = modelShikuangPtr->value();
    } else if(voidPtr == modelBeethovenPtr) {
        config.model_beethoven = modelBeethovenPtr->value();
    } else {
        SPDLOG_DEBUG("没有匹配元素");
    }
}

static void chooseDirectoryCallback(Fl_Widget* widget, void* voidPtr) {
    lifuren::directoryChooser(widget, voidPtr, "选择目录");
    auto& config = lifuren::config::CONFIG;
    if(voidPtr == tmpPtr) {
        config.tmp = tmpPtr->value();
    } else if(voidPtr == outputPtr) {
        config.output = outputPtr->value();
    } else {
        SPDLOG_DEBUG("没有匹配元素");
    }
}
