#include "../header/DLibAll.hpp"

#ifdef WIN32

#include <cmath>

#include "dlib/gui_widgets.h"
#include "dlib/image_transforms.h"

void lifuren::test3DPointCloud() {
    dlib::rand rand;
    std::vector<dlib::perspective_window::overlay_dot> points;
    for (double i = 0; i < 20; i += 0.001) {
        dlib::vector<double> val(sin(i), cos(i), i / 4);
        dlib::vector<double> temp(
            rand.get_random_gaussian(),
            rand.get_random_gaussian(),
            rand.get_random_gaussian()
        );
        val += temp / 20;
        dlib::rgb_pixel color = dlib::colormap_jet(i, 0, 20);
        points.push_back(dlib::perspective_window::overlay_dot(val, color));
        // dlib::vector<double> val(sin(i), 0, 0);
        // dlib::vector<double> val(cos(i), 0, 0);
        // dlib::vector<double> val(sin(i) + cos(i), 0, 0);
        // dlib::vector<double> val(sin(i), cos(i), i / 4);
        // dlib::rgb_pixel color = dlib::colormap_jet(i, 0, 20);
        // points.push_back(dlib::perspective_window::overlay_dot(val, color));
    }
    dlib::perspective_window win;
    win.set_title("3DPointCloud");
    win.set_size(512, 512);
    win.add_overlay(points);
    win.wait_until_closed();
}

#endif