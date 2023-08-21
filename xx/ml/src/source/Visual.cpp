#include "../header/Visual.hpp"

void lifuren::dots(std::vector<double>* x, std::vector<double>* y, int length, int xLength, int yLength, std::vector<double>* dx, std::vector<double>* dy, const char* title, int width, int height) {
    mglGraph graph(0, width, height);
    std::vector z(length, 0);
    mglData xx(*x);
    mglData yy(*y);
    mglData zz(z);
    graph.Title(title);
    graph.SetRanges(0, xLength, 0, yLength);
    graph.Dots(xx, yy, zz, "r1");
    if(dx != nullptr && dy != nullptr) {
        mglData xxd(*dx);
        mglData yyd(*dy);
        graph.Dots(xxd, yyd, zz, "g1");
    }
    graph.Axis();
    graph.Grid();
    cv::Mat mat(graph.GetHeight(), graph.GetWidth(), CV_8UC3);
    mat.data = (uchar*) graph.GetRGB();
    cv::imshow(title, mat);
    cv::waitKey();
}