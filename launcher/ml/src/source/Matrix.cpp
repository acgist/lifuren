#include "../header/MLPack.hpp"

void lifuren::matrix() {
    arma::mat a = arma::randu<arma::mat>(100, 5);
    LOG(INFO) << a;
    arma::mat b(50, 2, arma::fill::randu);
    LOG(INFO) << b;
}