#include "matplot/matplot.h"

int main() {
    auto plot = matplot::figure();
    plot->name("acgist");
    plot->number_title(false);
    matplot::hold(matplot::on);
    std::vector<double> x = matplot::linspace(0, 2 * matplot::pi);
    std::vector<double> y = matplot::transform(x, [](auto x) {
        return sin(x);
    });
    matplot::plot(x, y, "-");
    matplot::plot(x, matplot::transform(y, [](auto y) {
        return -y;
    }), "--r");
    matplot::show(plot);
    return 0;
}
