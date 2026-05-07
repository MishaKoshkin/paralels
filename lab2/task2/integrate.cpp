#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>

using namespace std;

const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

double func(double x) {
    return exp(-x * x);
}

double integrate_omp(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;
    
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        
        double sumloc = 0.0;
        
        for (int i = lb; i <= ub; i++) {
            sumloc += func(a + h * (i + 0.5));
        }
        
        #pragma omp atomic
        sum += sumloc;
    }
    
    sum *= h;
    return sum;
}

int main() {
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int num_threads_count = 8;
    
    ofstream outfile("integrate_benchmark.txt");
    outfile << "# Потоков Время(сек) Ускорение" << endl;
    
    double base_time = 0.0;
    
    cout << "Численное интегрирование методом прямоугольников" << endl;
    cout << "nsteps = " << nsteps << ", интервал [" << a << ", " << b << "]" << endl;
    
    for (int t = 0; t < num_threads_count; t++) {
        int num_threads = threads[t];
        omp_set_num_threads(num_threads);
        
        double start_time = omp_get_wtime();
        double result = integrate_omp(func, a, b, nsteps);
        double end_time = omp_get_wtime();
        
        double time = end_time - start_time;
        
        if (num_threads == 1) {
            base_time = time;
        }
        
        double speedup = base_time / time;
        
        cout << "Потоков: " << num_threads 
             << " | Время: " << time 
             << " сек | Результат: " << result
             << " | Ускорение: " << speedup << "x" << endl;
        
        outfile << num_threads << " " << time << " " << speedup << endl;
    }
    
    outfile.close();
    cout << "Результаты сохранены в integrate_benchmark.txt" << endl;
    
    return 0;
}