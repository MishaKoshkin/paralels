#include <iostream>
#include <iomanip>
#include <cmath>

#ifdef USE_FLOAT
using type = float;
#else
using type = double;
#endif

double normalize_angle(type x) {
    constexpr type PI = 3.14159265358979323846;
    constexpr type TWO_PI = 2.0 * PI;
    x = std::fmod(x + 3.0 * PI, TWO_PI);
    if (x > PI)  x -= TWO_PI;
    if (x < -PI) x += TWO_PI;
    return x;
}

// sin(x) через ряд Тейлора
double my_sin_taylor(type x, int terms = 8) {
    if (terms < 1) terms = 1;

    x = normalize_angle(x);

    // Приведение к [-π/2, π/2] + определение знака
    bool negative = x < 0;
    if (negative) x = -x;
    if (x > 1.5707963267948966) {  // > π/2
        x = 3.14159265358979323846 - x;
    }

    type sum = x;
    type term = x;
    type x2 = x * x;

    // Начинаем с n=1 (второй член ряда)
    for (int n = 1; n < terms; ++n) {
        term = -term * x2 / ((2 * n) * (2 * n + 1));
        sum += term;
    }

    return negative ? -sum : sum;
}

int main()
{
    const int N = 10000000;
    const type PI = 3.14159265358979323846;
    const type step = (2.0 * PI) / (N - 1.0);

    double sum = 0.0;
    double angle = 0.0;

    for (int i = 0; i < N; ++i)
    {
        sum += my_sin_taylor(angle, 10);
        angle += step;
    }


    std::cout << std::fixed << std::setprecision(12);
    std::cout << "N = " << N << "\n";
    std::cout << "SUM = " << sum << "\n";

    return 0;
}