#include "../../header/MLPack.hpp"

void testPlus() {
    // arma::mat a = arma::randu<arma::mat>(2, 3);
    // arma::mat b = arma::randu<arma::mat>(2, 3);
    // arma::mat c = arma::randu<arma::mat>(2, 2);
    double array[] = {
        1, 2, 3,
        4, 5, 6
    };
    double square[] = {
        1, 2,
        3, 4
    };
    arma::mat a(array, 3, 2);
    arma::mat b(array, 3, 2);
    arma::mat c(square, 2, 2);
    LOG(INFO) << "a = " << std::endl << a;
    LOG(INFO) << "b = " << std::endl << b;
    LOG(INFO) << "c = " << std::endl << c;
    LOG(INFO) << "a + b = "  << std::endl << (a + b);
    LOG(INFO) << "a - b = "  << std::endl << (a - b);
    LOG(INFO) << "a + 1 = "  << std::endl << (a + 1);
    LOG(INFO) << "a * 2 = "  << std::endl << (a * 2);
    LOG(INFO) << "a -> t = " << std::endl << a.t();
    LOG(INFO) << "c -> i = " << std::endl << c.i();
    LOG(INFO) << "a -> as_col = " << std::endl << a.as_col();
    LOG(INFO) << "a -> as_row = " << std::endl << a.as_row();
    LOG(INFO) << "min = "  << a.min();
    LOG(INFO) << "max = "  << a.max();
    LOG(INFO) << "size = " << a.size();
    LOG(INFO) << "n_cols = " << a.n_cols;
    LOG(INFO) << "n_rows = " << a.n_rows;
    LOG(INFO) << "sum = "   << std::endl << arma::sum(a);
    LOG(INFO) << "mean = "  << std::endl << arma::mean(a);
    // 乘积
    LOG(INFO) << "prod = "  << std::endl << arma::prod(a);
    LOG(INFO) << "trace = " << arma::trace(a);
    LOG(INFO) << std::endl << std::endl << std::endl << std::endl;
}

void testFill() {
    LOG(INFO) << std::endl << arma::randu<arma::mat>(3, 5);
    LOG(INFO) << std::endl << arma::mat(5, 3, arma::fill::randu);
    LOG(INFO) << "eye   = " << std::endl << arma::mat(2, 2, arma::fill::eye);
    LOG(INFO) << "ones  = " << std::endl << arma::mat(2, 2, arma::fill::ones);
    LOG(INFO) << "zeros = " << std::endl << arma::mat(2, 2, arma::fill::zeros);
    LOG(INFO) << "randn = " << std::endl << arma::mat(2, 2, arma::fill::randn);
    LOG(INFO) << "randu = " << std::endl << arma::mat(2, 2, arma::fill::randu);
    LOG(INFO) << std::endl << arma::randg<arma::mat>(2, 2);
    LOG(INFO) << std::endl << arma::randi<arma::mat>(2, 2);
    LOG(INFO) << std::endl << arma::randn<arma::mat>(2, 2);
    LOG(INFO) << std::endl << arma::randu<arma::mat>(2, 2);
}

void testMult() {
    arma::mat a = arma::randu(3, 5);
    LOG(INFO) << std::endl << a;
    LOG(INFO) << std::endl << (a + 4);
    LOG(INFO) << std::endl << (a - 4);
    LOG(INFO) << std::endl << (a * 4);
    LOG(INFO) << std::endl << (a / 2);
}

void testTransfer() {
    double array[] = {
        // 1, 2, 3, 4,
        // 1, 2, 3, 4,
        // 1, 1, 1, 1,
        1,  2,  3,  4,
        20, 21, 22, 23,
        11, 12, 13, 14,
    };
    // arma::mat a(array, 3, 4);
    arma::mat a(array, 4, 3);
    LOG(INFO) << std::endl << a;
    LOG(INFO) << std::endl << a + a;
    double flat[] = { 1, 1, 1 };
    // double flat[] = { 1, 2, 3 };
    arma::rowvec b(flat, 3);
    LOG(INFO) << std::endl << b;
    double mult[] = { 1, 1, 1, 2 };
    arma::rowvec c(mult, 4);
    LOG(INFO) << std::endl << c;
    LOG(INFO) << std::endl << arma::sum(a);
    LOG(INFO) << std::endl << arma::sum(a, 1);
    LOG(INFO) << std::endl << arma::sum(a.t());
}

// 点积、内积、数量积、标量积
void testDotProduct() {
    // 向量
    double x[] = { 1, 1, 1 };
    arma::rowvec a(x, 3);
    double y[] = { 3, 2, 1 };
    arma::rowvec b(y, 3);
    LOG(INFO) << a;
    LOG(INFO) << b;
    LOG(INFO) << a * b.t();
    LOG(INFO) << arma::dot(a, b);
    LOG(INFO) << arma::dot(a, b.t());
    // 矩阵
    double ca[] = { 1, 3, 2, 4 };
    double da[] = { 5, 7, 6, 8 };
    arma::mat c(ca, 2, 2);
    arma::mat d(da, 2, 2);
    LOG(INFO) << std::endl << c;
    LOG(INFO) << std::endl << d;
    LOG(INFO) << std::endl << (c * d);
    LOG(INFO) << std::endl << (c % d);
    LOG(INFO) << std::endl << (c / d);
    LOG(INFO) << std::endl << (c + d);
    LOG(INFO) << std::endl << (c - d);
    LOG(INFO) << arma::dot(c, d);
    double ea[] = { 1, 1, 2, 2, 3, 3 };
    double fa[] = { 1, 2, 3 };
    arma::mat e(ea, 2, 3);
    arma::mat f(fa, 1, 3);
    LOG(INFO) << std::endl << e;
    LOG(INFO) << std::endl << f;
    LOG(INFO) << std::endl << (e * f.t());
}

// 叉积、叉乘、向量积
void testCrossProduct() {
    double x[] = {
        1, 1, 1,
        1, 1, 1
    };
    arma::mat a(x, 3, 2);
    double y[] = {
        3, 2,
        1, 3,
        2, 1
    };
    arma::mat b(y, 2, 3);
    LOG(INFO) << std::endl << a;
    LOG(INFO) << std::endl << b;
    LOG(INFO) << std::endl << (a * b);
    LOG(INFO) << std::endl << (a % b.t());
    double xx[] = { 1, 1, 1 };
    arma::mat xa(xx, 1, 3);
    // arma::rowvec xa(xx, 3);
    double yy[] = { 3, 2, 1 };
    arma::mat yb(yy, 1, 3);
    // arma::rowvec yb(yy, 3);
    LOG(INFO) << std::endl << arma::cross(xa, yb);
    LOG(INFO) << std::endl << arma::cross(yb, xa);
}

// 外积、张量积
void testOuterProduct() {
    double x[] = { 1, 1, 1 };
    arma::mat a(x, 1, 3);
    arma::rowvec aa(x, 3);
    double y[] = { 3, 2, 1 };
    arma::mat b(y, 3, 1);
    arma::rowvec bb(y, 3);
    LOG(INFO) << std::endl << a;
    LOG(INFO) << std::endl << b;
    LOG(INFO) << std::endl << aa;
    LOG(INFO) << std::endl << bb;
    LOG(INFO) << std::endl << (a % b.t());
    LOG(INFO) << std::endl << (a * b);
    // 克罗内克积
    LOG(INFO) << std::endl << arma::kron(a, b);
    LOG(INFO) << std::endl << arma::kron(a, b.t());
}

void testSchur() {
    // 元素积、舒尔积、逐项积、哈达玛积
    arma::mat X(2, 2, arma::fill::randu);
    arma::mat U;
    arma::mat S;
    arma::schur(U, S, X);
    LOG(INFO) << std::endl << X;
    LOG(INFO) << std::endl << U;
    LOG(INFO) << std::endl << S;
}

void testSvd() {
    arma::mat X(5, 5, arma::fill::randu);
    arma::vec S;
    arma::mat U;
    arma::mat V;
    arma::svd(U, S, V, X);
    LOG(INFO) << std::endl << X;
    LOG(INFO) << std::endl << S;
    LOG(INFO) << std::endl << U;
    LOG(INFO) << std::endl << V;
}

void testResize() {
    double array[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 5,
        6, 7, 8,
    };
    arma::mat mat(array, 3, 4);
    LOG(INFO) << std::endl << mat << std::endl;
    LOG(INFO) << std::endl << mat.t() << std::endl;
    LOG(INFO) << std::endl << mat.st() << std::endl;
    LOG(INFO) << std::endl << mat << std::endl;
    mat.shed_col(0);
    LOG(INFO) << std::endl << mat << std::endl;
    mat.shed_row(0);
    LOG(INFO) << std::endl << mat << std::endl;
    mat.resize(2, 2);
    LOG(INFO) << std::endl << mat << std::endl;
}

void testShuffle() {
    double array[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 5,
        6, 7, 8,
    };
    arma::mat mat(array, 3, 4);
    LOG(INFO) << std::endl << mat << std::endl;
    mat = arma::shuffle(mat, 0);
    LOG(INFO) << std::endl << mat << std::endl;
    mat = arma::shuffle(mat, 0);
    LOG(INFO) << std::endl << mat << std::endl;
    mat = arma::shuffle(mat, 0);
    LOG(INFO) << std::endl << mat << std::endl;
}

void testInit() {
    arma::mat mat = {
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
    };
    double array[] = {
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
    };
    LOG(INFO) << std::endl << arma::mat(array, 3, 4) << std::endl;
    LOG(INFO) << std::endl << arma::mat(array, 4, 3) << std::endl;
    LOG(INFO) << std::endl << mat << std::endl;
    arma::mat tail = mat.tail_cols(2);
    LOG(INFO) << std::endl << tail << std::endl;
    LOG(INFO) << std::endl << mat.tail_cols(2) << std::endl;
    LOG(INFO) << std::endl << mat.tail_rows(2) << std::endl;
}

void testEquals() {
    arma::mat source = {
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
    };
    arma::mat target = {
        { 1, 2, 3, 4 },
        { 1, 2, 1, 4 },
        { 1, 2, 3, 4 },
    };
    LOG(INFO) << std::endl << source << std::endl;
    LOG(INFO) << std::endl << target << std::endl;
    double accu = arma::accu(source == target);
    LOG(INFO) << std::endl << (source == target) << std::endl;
    LOG(INFO) << std::endl << arma::sum(source) << std::endl;
    LOG(INFO) << std::endl << arma::accu(source) << std::endl;
    LOG(INFO) << std::endl << arma::accu(target) << std::endl;
    LOG(INFO) << std::endl << arma::affmul(source, target) << std::endl;
    LOG(INFO) << accu << std::endl;
}

void lifuren::testMLPackMatrix() {
    testPlus();
    // testFill();
    // testMult();
    // testTransfer();
    // testDotProduct();
    // testCrossProduct();
    // testOuterProduct();
    // testSchur();
    // testSvd();
    // testResize();
    // testShuffle();
    // testInit();
    // testEquals();
}