#include <cmath>
#include <iomanip>
#include <iostream>

int main(int argc, char const *argv[]) {
    std::cout
    <<
    std::fixed
    // std::ios::fixed
    << std::setprecision(2);
    float a = 100.1234F;
    a = a++ + 1;
    a++;
    std::cout << a << std::endl;
    double b = 1.000001;
    double c = 1.0000011;
    double d = b + c;
    std::cout << std::setw(10) << d << std::endl;
    // int e = 0 / 0;
    // std::cout << e << std::endl;
    double da { 1.5 }, db {}; // db = 0
    double dc { da / db};
    std::cout << dc << std::endl;
    double dd { 1.5 }, de {}; // de = 0
    double df { dd / de};
    std::cout << df << std::endl;
    // std::cout << 1   / 0 << std::endl;
    // std::cout << 1.2 / 0 << std::endl;
    std::cout << std::left << 1 << std::endl;
    std::cout << std::right << 1 << std::endl;
    std::cout << "dec" << 16 << std::endl;
    std::cout << "hex" << std::hex << 16 << std::endl;
    std::cout << std::setw(20) << "hex?" << 16 << std::endl;
    std::cout << std::setw(20) << std::left << "hex?" << 16 << std::endl;
    unsigned int ua = 10U;
    int ub = 20;
    std::cout << std::dec << ua - ub << std::endl;
    int uc = ua;
    std::cout << uc << std::endl;
    std::cout << std::dec << uc - ub << std::endl;
    unsigned int ud = ub;
    std::cout << ud << std::endl;
    return 0;
}
