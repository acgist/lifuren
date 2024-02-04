#include "../../header/MLPack.hpp"

void lifuren::testMLPackLoadFile(const char* path) {
    arma::mat mat;
    mlpack::data::DatasetInfo mapping;
    mapping.Type(0) = mlpack::data::Datatype::numeric;
    mapping.Type(1) = mlpack::data::Datatype::numeric;
    mapping.Type(2) = mlpack::data::Datatype::numeric;
    mapping.Type(3) = mlpack::data::Datatype::numeric;
    mapping.Type(4) = mlpack::data::Datatype::categorical;
    mlpack::data::Load(path, mat, mapping);
    LOG(INFO) << mapping.Type(0);
    LOG(INFO) << mapping.Type(1);
    LOG(INFO) << mapping.Type(2);
    LOG(INFO) << mapping.Type(3);
    LOG(INFO) << mapping.Type(4);
    LOG(INFO) << "原始数据" << std::endl << mat.t();
    // 删除序号
    // mat.shed_col(0);
    // 删除头部
    // mat.shed_row(0);
    // 修改序号
    // mat.row(4) -= 1;
    LOG(INFO) << std::endl << mat.t();
    LOG(INFO) << "数据大小：" << mat.n_cols << " * " << mat.n_rows << std::endl;
    for(int col = mat.n_cols - 1; col >= 0; --col) {
        if(mat.col(col)[4] > 1) {
            mat.shed_col(col);
        }
    }
    LOG(INFO) << std::endl << mat.t();
}
