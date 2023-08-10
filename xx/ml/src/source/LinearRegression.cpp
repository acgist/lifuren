#include "../header/DLibAll.hpp"

#include <vector>
#include <random>
#include <dlib/svm.h>
#include <dlib/matrix.h>
#include "dlib/gui_widgets.h"

namespace lifuren {

namespace ml {

    typedef dlib::matrix<double, 1, 1> sample_type;
    // typedef dlib::matrix<double, 2, 1> sample_type;
    typedef dlib::radial_basis_kernel<sample_type> kernel_type;

    void setTrainDataSet(std::vector<dlib::matrix<double>>& x, std::vector<double>& y) {
        // dlib::rand rand;
        for (int i = 0; i < 100; i++) {
            double v = std::rand() % 100;
            sample_type xx;
            xx(0) = v;
            x.push_back(xx);
            double yy = v * 12 + 8 + std::rand() % 10;
            y.push_back(yy);
        }
    }

    void testLinearRegression() {
        std::vector<dlib::matrix<double>> x;
        std::vector<double> y;
        setTrainDataSet(x, y);
        dlib::krr_trainer<kernel_type> trainer;
        trainer.set_kernel(kernel_type());
        dlib::decision_function<kernel_type> predict = trainer.train(x, y);
        std::vector<dlib::matrix<double>> new_x;
        std::vector<dlib::image_window::overlay_circle> points;
        for(int index = 0; index < x.size(); index++) {
            double xx = x.at(index);
            double yy = y.at(index);
            points.push_back(dlib::image_window::overlay_circle(dlib::point(yy, xx), 1, dlib::rgb_pixel(0, 0, 0)));
        }
        for (std::vector<dlib::matrix<double>>::iterator::value_type& v : x) {
            double xx = *v.begin();
            double prediction = predict(v);
            points.push_back(dlib::image_window::overlay_circle(dlib::point(prediction, xx), 1, dlib::rgb_pixel(255, 0, 0)));
        }
        dlib::image_window win;
        // dlib::perspective_window win;
        win.set_title("LinearRegression");
        win.set_size(512, 512);
        win.add_overlay(points);
        win.set_background_color(255, 255, 255);
        // win.add_overlay()
        // win.add_overlay(points, dlib::rgb_pixel(255, 0, 0));
        win.wait_until_closed();
    }
}

}