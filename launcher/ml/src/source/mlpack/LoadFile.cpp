#include "../../header/MLPack.hpp"

void lifuren::testLoadFile() {
    arma::mat mat;
    mlpack::data::DatasetInfo mapping;
    mlpack::data::Load("D:\\tmp\\ml\\iris.csv", mat, mapping);
    mat.shed_col(0);
    mat.shed_row(0);
    mat.row(4) -= 1;
    LOG(INFO) << std::endl << mat.t() << std::endl;
    LOG(INFO) << mat.n_cols << " = " << mat.n_rows << std::endl;
    for(int col = mat.n_cols - 1; col >= 0; --col) {
        if(mat.col(col)[4] > 1) {
            mat.shed_col(col);
        }
    }
    LOG(INFO) << std::endl << mat.t() << std::endl;
}
