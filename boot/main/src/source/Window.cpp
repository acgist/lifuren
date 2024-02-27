#include "../header/Window.hpp"

const char* lifuren::fileChooser(const char* title, const char* directory, const char* filter) {
    Fl_Native_File_Chooser chooser(Fl_Native_File_Chooser::BROWSE_FILE);
    chooser.title(title);
    chooser.filter(filter);
    chooser.directory(directory);
    const int code = chooser.show();
    switch(code) {
        case 0: {
            const char* filename = chooser.filename();
            SPDLOG_DEBUG("文件选择成功：{} - {}", title, filename);
            return filename;
        }
        default:
            SPDLOG_DEBUG("文件选择失败：{} - {}", title, code);
            return "";
    }
}

const char* lifuren::directoryChooser(const char* title, const char* directory) {
    Fl_Native_File_Chooser chooser(Fl_Native_File_Chooser::BROWSE_DIRECTORY);
    chooser.title(title);
    chooser.directory(directory);
    const int code = chooser.show();
    switch(code) {
        case 0: {
            const char* filename = chooser.filename();
            SPDLOG_DEBUG("目录选择成功：{} - {}", title, filename);
            return filename;
        }
        default:
            SPDLOG_DEBUG("目录选择失败：{} - {}", title, code);
            return "";
    }
}

lifuren::Fl_Input_Directory_Chooser::Fl_Input_Directory_Chooser(
    int x,
    int y,
    int width,
    int height,
    const char* title,
    const char* directory
) :
    Fl_Input(x, y, width, height),
    title(title),
    directory(directory)
{
    this->label(title);
}

lifuren::Fl_Input_Directory_Chooser::~Fl_Input_Directory_Chooser() {
}

int lifuren::Fl_Input_Directory_Chooser::handle(int event) {
    if(event == FL_LEFT_MOUSE) {
        const char* filename = directoryChooser(this->title);
        this->value(filename);
        return 0;
    }
    return Fl_Input::handle(event);
}
