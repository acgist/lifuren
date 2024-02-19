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
    LOG(INFO) << "a + b = " << std::endl << (a + b);
    LOG(INFO) << "a - b = " << std::endl << (a - b);
    LOG(INFO) << "a + 1 = " << std::endl << (a + 1);
    LOG(INFO) << "a * 2 = " << std::endl << (a * 2);
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
    LOG(INFO) << "eye   = " << std::endl << arma::mat(2, 2, arma::fill::eye);
    LOG(INFO) << "ones  = " << std::endl << arma::mat(2, 2, arma::fill::ones);
    LOG(INFO) << "zeros = " << std::endl << arma::mat(2, 2, arma::fill::zeros);
    LOG(INFO) << "randn = " << std::endl << arma::mat(2, 2, arma::fill::randn);
    LOG(INFO) << "randu = " << std::endl << arma::mat(2, 2, arma::fill::randu);
    LOG(INFO) << "randg = " << std::endl << arma::randg<arma::mat>(2, 3);
    LOG(INFO) << "randi = " << std::endl << arma::randi<arma::mat>(2, 3);
    LOG(INFO) << "randn = " << std::endl << arma::randn<arma::mat>(2, 3);
    LOG(INFO) << "randu = " << std::endl << arma::randu<arma::mat>(2, 3);
    LOG(INFO) << "randperm = " << std::endl << arma::randperm<arma::mat>(10);
    LOG(INFO) << std::endl << std::endl << std::endl << std::endl;
}

void testFour() {
    arma::mat a = arma::randu(3, 5);
    LOG(INFO) << "a = " << std::endl << a;
    LOG(INFO) << "a + 4 = " << std::endl << (a + 4);
    LOG(INFO) << "a - 4 = " << std::endl << (a - 4);
    LOG(INFO) << "a * 4 = " << std::endl << (a * 4);
    LOG(INFO) << "a / 2 = " << std::endl << (a / 2);
    LOG(INFO) << std::endl << std::endl << std::endl << std::endl;
}

void testSelf() {
    double array[] = {
        1,  2,  3,  4,
        11, 12, 13, 14,
        21, 22, 23, 24,
    };
    // arma::mat a(array, 3, 4);
    arma::mat a(array, 4, 3);
    LOG(INFO) << "a = " << std::endl << a;
    LOG(INFO) << "a + a = " << std::endl << a + a;
    double arrayb[] = { 1, 1, 1 };
    arma::rowvec b(arrayb, 3);
    LOG(INFO) << "b = " << std::endl << b;
    double arrayc[] = { 1, 2, 3, 4 };
    arma::rowvec c(arrayc, 4);
    LOG(INFO) << "c = " << std::endl << c;
    LOG(INFO) << "a col sum = " << std::endl << arma::sum(a);
    LOG(INFO) << "a row sum = " << std::endl << arma::sum(a, 1);
    LOG(INFO) << "a.t col sum = " << std::endl << arma::sum(a.t());
    LOG(INFO) << std::endl << std::endl << std::endl << std::endl;
}

// 张量积
// 克罗内克积
// 叉积、叉乘、外积、向量积
// 点积、内积、数量积、标量积
// 元素积、舒尔积、逐项积、哈达玛积
void testProduct() {
    // 向量
    double x[] = { 1, 1, 1 };
    arma::rowvec a(x, 3);
    double y[] = { 3, 2, 1 };
    arma::rowvec b(y, 3);
    LOG(INFO) << "a =" << std::endl << a;
    LOG(INFO) << "b =" << std::endl << b;
    LOG(INFO) << "a * b.t =" << std::endl << a * b.t();
    LOG(INFO) << "a dot b =" << std::endl << arma::dot(a, b);
    LOG(INFO) << "a dot b.t =" << std::endl << arma::dot(a, b.t());
    // 矩阵
    double ca[] = { 1, 3, 2, 4 };
    double da[] = { 5, 7, 6, 8 };
    arma::mat c(ca, 2, 2);
    arma::mat d(da, 2, 2);
    LOG(INFO) << "c =" << std::endl << c;
    LOG(INFO) << "d =" << std::endl << d;
    LOG(INFO) << "c * d =" << std::endl << (c * d);
    LOG(INFO) << "c % d =" << std::endl << (c % d);
    LOG(INFO) << "c / d =" << std::endl << (c / d);
    LOG(INFO) << "c + d =" << std::endl << (c + d);
    LOG(INFO) << "c - d =" << std::endl << (c - d);
    LOG(INFO) << "c dot d =" << std::endl << arma::dot(c, d);
    // 向量矩阵
    double ea[] = { 1, 1, 2, 2, 3, 3 };
    double fa[] = { 1, 2, 3 };
    arma::mat e(ea, 2, 3);
    arma::mat f(fa, 1, 3);
    LOG(INFO) << "e =" << std::endl << e;
    LOG(INFO) << "f =" << std::endl << f;
    LOG(INFO) << "e * f.t =" << std::endl << (e * f.t());
    // 大小不等矩阵
    double ga[] = {
        1, 1, 1,
        1, 1, 1
    };
    arma::mat g(ga, 3, 2);
    double ha[] = {
        3, 2,
        1, 3,
        2, 1
    };
    arma::mat h(ha, 2, 3);
    LOG(INFO) << "g =" << std::endl << g;
    LOG(INFO) << "h =" << std::endl << h;
    LOG(INFO) << "g * h =" << std::endl << (g * h);
    LOG(INFO) << "h * g =" << std::endl << (h * g);
    LOG(INFO) << "g % h.t =" << std::endl << (g % h.t());
    LOG(INFO) << "h % g.h =" << std::endl << (h % g.t());
    // 叉乘
    double xx[] = { 1, 1, 1 };
    arma::mat xa(xx, 1, 3);
    // arma::rowvec xa(xx, 3);
    double yy[] = { 3, 2, 1 };
    arma::mat yb(yy, 1, 3);
    // arma::rowvec yb(yy, 3);
    LOG(INFO) << "xa =" << std::endl << xa;
    LOG(INFO) << "yb =" << std::endl << yb;
    LOG(INFO) << "xa cross yb =" << std::endl << arma::cross(xa, yb);
    LOG(INFO) << "yb cross xa =" << std::endl << arma::cross(yb, xa);
    // 克罗内克积
    double ooa[] = { 1, 1, 1 };
    arma::mat oa(ooa, 1, 3);
    arma::rowvec oaa(ooa, 3);
    double oob[] = { 3, 2, 1 };
    arma::mat ob(oob, 3, 1);
    arma::rowvec obb(oob, 3);
    LOG(INFO) << "oa =" << std::endl << oa;
    LOG(INFO) << "ob =" << std::endl << ob;
    LOG(INFO) << "oaa =" << std::endl << oaa;
    LOG(INFO) << "obb =" << std::endl << obb;
    LOG(INFO) << "oa % ob.t =" << std::endl << (oa % ob.t());
    LOG(INFO) << "oa * ob =" << std::endl << (oa * ob);
    LOG(INFO) << "oa kron ob =" << std::endl << arma::kron(oa, ob);
    LOG(INFO) << "oa kron ob.t =" << std::endl << arma::kron(oa, ob.t());
    LOG(INFO) << std::endl << std::endl << std::endl << std::endl;
}

void testSvd() {
    arma::mat X(5, 5, arma::fill::randu);
    arma::vec S;
    arma::mat U;
    arma::mat V;
    arma::svd(U, S, V, X);
    LOG(INFO) << "X =" << std::endl << X;
    LOG(INFO) << "S =" << std::endl << S;
    LOG(INFO) << "U =" << std::endl << U;
    LOG(INFO) << "V =" << std::endl << V;
    LOG(INFO) << std::endl << std::endl << std::endl << std::endl;
}

void testSchur() {
    arma::mat X(2, 2, arma::fill::randu);
    arma::mat U;
    arma::mat S;
    arma::schur(U, S, X);
    LOG(INFO) << "X =" << std::endl << X;
    LOG(INFO) << "U =" << std::endl << U;
    LOG(INFO) << "S =" << std::endl << S;
    LOG(INFO) << std::endl << std::endl << std::endl << std::endl;
}

void testModify() {
    double array[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 5,
        6, 7, 8,
    };
    arma::mat mat(array, 3, 4);
    LOG(INFO) << "mat =" << std::endl << mat;
    LOG(INFO) << "mat.t =" << std::endl << mat.t();
    LOG(INFO) << "mat.st =" << std::endl << mat.st();
    LOG(INFO) << "mat =" << std::endl << mat;
    mat.shed_col(0);
    LOG(INFO) << "mat.shed_col =" << std::endl << mat;
    mat.shed_row(0);
    LOG(INFO) << "mat.shed_row =" << std::endl << mat;
    mat.resize(2, 2);
    LOG(INFO) << "mat.resize =" << std::endl << mat;
    LOG(INFO) << std::endl << std::endl << std::endl << std::endl;
}

void testShuffle() {
    double array[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 5,
        6, 7, 8,
    };
    arma::mat mat(array, 3, 4);
    LOG(INFO) << "mat =" << std::endl << mat;
    mat = arma::shuffle(mat);
    LOG(INFO) << "mat.shuffle =" << std::endl << mat;
    mat = arma::shuffle(mat);
    LOG(INFO) << "mat.shuffle =" << std::endl << mat;
    mat = arma::shuffle(mat, 0);
    LOG(INFO) << "mat.shuffle row =" << std::endl << mat;
    mat = arma::shuffle(mat, 0);
    LOG(INFO) << "mat.shuffle row =" << std::endl << mat;
    mat = arma::shuffle(mat, 1);
    LOG(INFO) << "mat.shuffle col =" << std::endl << mat;
    mat = arma::shuffle(mat, 1);
    LOG(INFO) << "mat.shuffle col =" << std::endl << mat;
    LOG(INFO) << std::endl << std::endl << std::endl << std::endl;
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
    LOG(INFO) << "mat 3*4 =" << std::endl << arma::mat(array, 3, 4);
    LOG(INFO) << "mat 4*3 =" << std::endl << arma::mat(array, 4, 3);
    LOG(INFO) << "mat =" << std::endl << mat;
    LOG(INFO) << "mat tail cols =" << std::endl << mat.tail_cols(2);
    LOG(INFO) << "mat tail rows =" << std::endl << mat.tail_rows(2);
    LOG(INFO) << std::endl << std::endl << std::endl << std::endl;
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
    LOG(INFO) << "source =" << std::endl << source;
    LOG(INFO) << "target =" << std::endl << target;
    LOG(INFO) << "source == target =" << std::endl << (source == target);
    LOG(INFO) << "accu(source == target) =" << arma::accu(source == target);
    LOG(INFO) << "affmul(source, target) =" << std::endl << arma::affmul(source, target);
    LOG(INFO) << "sum(source) =" << std::endl << arma::sum(source);
    LOG(INFO) << "accu(source) =" << std::endl << arma::accu(source);
    LOG(INFO) << "accu(target) =" << std::endl << arma::accu(target);
}

void lifuren::testMLPackMatrix() {
    testPlus();
    testFill();
    testFour();
    testSelf();
    testProduct();
    testSvd();
    testSchur();
    testModify();
    testShuffle();
    testInit();
    testEquals();
}
