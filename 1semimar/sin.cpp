#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#ifdef USE_DOUBLE
using real_t = double;
#else
using real_t = float;
#endif

int main(){
    const std::size_t N = 10000000;
    const real_t two_pi = static_cast<real_t>(2.0 * M_PI);

    std::vector<real_t> data(N);

    real_t sum = 0;

    for (std::size_t i = 0; i < N; ++i){
        data[i] = std::sin(two_pi * static_cast<real_t>(i) / static_cast<real_t>(N));
        sum += data[i];
    }
    std::cout << std::fixed << std::setprecision(15)<< sum<< '\n';
    return 0;
}