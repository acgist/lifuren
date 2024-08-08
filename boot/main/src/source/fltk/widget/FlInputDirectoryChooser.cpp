#include "lifuren/FLTK.hpp"
#include "lifuren/FLTKWidget.hpp"

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
        const std::string filename = lifuren::directoryChooser(this->title);
        if(filename.empty()) {
            return 0;
        }
        this->value(filename.c_str());
        this->do_callback();
        return 0;
    }
    return Fl_Input::handle(event);
}
