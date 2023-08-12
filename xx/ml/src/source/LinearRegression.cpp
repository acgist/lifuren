#include "../header/DLibAll.hpp"

#include <vector>
#include <random>

#include "mgl2/mgl.h"
#include "dlib/svm.h"
#include "dlib/matrix.h"
#include "opencv2/opencv.hpp"

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
        for(int index = 0; index < x.size(); index++) {
            double xx = x.at(index);
            double yy = y.at(index);
        }
        for (std::vector<dlib::matrix<double>>::iterator::value_type& v : x) {
            double xx = *v.begin();
            double prediction = predict(v);
        }

         mglGraph gr;
         gr.Title("MathGL Demo");
         gr.SetOrigin(0, 0);
         gr.SetRanges(0, 10, -2.5, 2.5);
         gr.FPlot("sin(1.7*2*pi*x) + sin(1.9*2*pi*x)", "r-2");
         gr.Axis();
         gr.Grid();
         gr.GetRGB();
         cv::Mat pic(gr.GetHeight(), gr.GetWidth(), CV_8UC3);
         pic.data = const_cast<uchar*>(gr.GetRGB());
        cv::imshow("test", pic);
        cv::waitKey();
        std::cout << "dd";
    }
}

}