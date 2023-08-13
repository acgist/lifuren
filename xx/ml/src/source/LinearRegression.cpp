#include "../header/DLibAll.hpp"

#include <vector>
#include <random>

#include "dlib/svm.h"
#include "dlib/matrix.h"

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
            double yy = v * 12 + 8 + std::rand() % 50;
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
        std::vector<double> xv(100);
        std::vector<double> yv(100);
        for(int index = 0; index < x.size(); index++) {
            xv[index] = x.at(index);
            yv[index] = y.at(index);
        }
        std::vector<double> pxv(100);
        std::vector<double> pyv(100);
        int index = 0;
        for (std::vector<dlib::matrix<double>>::iterator::value_type& v : x) {
            pxv[index] = *v.begin();
            pyv[index] = predict(v);
            index++;
        }
        lifuren::gg::dots(&xv, &yv, 100, 100, 1000);
        lifuren::gg::dots(&xv, &yv, 100, 100, 1000, &pxv, &pyv);
    }
    
}

}