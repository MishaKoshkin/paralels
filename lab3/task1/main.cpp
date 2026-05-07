#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstdlib>

using namespace std;
using namespace chrono;

class Timer {
    high_resolution_clock::time_point start;
public:
    Timer() { reset(); }
    void reset() { start = high_resolution_clock::now(); }
    double elapsed() { 
        return duration_cast<milliseconds>(high_resolution_clock::now() - start).count() / 1000.0;
    }
};

void fill_matrix_row(double* row, int n, int row_idx) {
    for (int j = 0; j < n; j++) {
        row[j] = sin(row_idx * 0.001) * cos(j * 0.001);
    }
}

void fill_matrix_part(double* A, int n, int start_row, int end_row) {
    for (int i = start_row; i < end_row; ++i) {
        fill_matrix_row(A + (long long)i * n, n, i);
    }
}

void fill_vector_part(double* vec, int start, int end) {
    for (int i = start; i < end; ++i) {
        vec[i] = exp(i * 0.0001);
    }
}

void mult_part(const double* A, const double* x, double* y, int n, int start, int end) {
    for (int i = start; i < end; i++) {
        double sum = 0;
        const double* row = A + (long long)i * n;
        for (int j = 0; j < n; j++) {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

int main(){
    
    cout << "\nИнформация о вычислительном узле:\n";
    system("lscpu | grep -E 'Model name|CPU\\(s\\)|Thread\\(s\\) per core|Core\\(s\\) per socket|Socket\\(s\\)|L3 cache'");
    cout << "\nМодель сервера:\n";
    system("cat /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo 'Недоступно'");
    cout << "\nNUMA ноды:\n";
    system("numactl --hardware 2>/dev/null | grep -A 10 'available' || echo 'numactl не установлен'");
    cout << "\nОперативная память:\n";
    system("free -h | grep Mem");
    cout << "\nОперационная система:\n";
    system("cat /etc/os-release | grep -E 'PRETTY_NAME|VERSION' | head -2");
    
    vector<int> sizes = {20000, 40000};
    vector<int> threads_list = {1, 2, 4, 7, 8, 16, 20, 40};
    
    ofstream data_file("benchmark_data.txt");
    if (!data_file.is_open()) {
        cerr << "Ошибка: не удалось создать файл benchmark_data.txt\n";
        return 1;
    }
    
    // Число потоков для инициализации 
    const int init_threads_count = 4;
    
    for (int size : sizes){
        cout << "\nРазмер матрицы: " << size << "x" << size << "\n";
        
        double* A = new double[(long long)size * size];
        double* x = new double[size];
        double* y = new double[size];
        
        // Параллельная инициализация матрицы
        cout << "  Генерация матрицы (параллельно, " << init_threads_count << " потока)..." << flush;
        {
            vector<thread> init_threads;
            int rows_per_thread = size / init_threads_count;
            for (int t = 0; t < init_threads_count; ++t) {
                int start = t * rows_per_thread;
                int end = (t == init_threads_count-1) ? size : (t+1) * rows_per_thread;
                init_threads.emplace_back(fill_matrix_part, A, size, start, end);
            }
            for (auto& th : init_threads) th.join();
        }
        cout << "готово\n";
        
        // Параллельная инициализация вектора (разобьём на те же части)
        cout << "  Генерация вектора (параллельно)..." << flush;
        {
            vector<thread> init_threads;
            int elems_per_thread = size / init_threads_count;
            for (int t = 0; t < init_threads_count; ++t) {
                int start = t * elems_per_thread;
                int end = (t == init_threads_count-1) ? size : (t+1) * elems_per_thread;
                init_threads.emplace_back(fill_vector_part, x, start, end);
            }
            for (auto& th : init_threads) th.join();
        }
        cout << "готово\n";
        
        double base_time = 0;
        
        for (int threads : threads_list) {
            cout << "  Потоков " <<  threads << ": " << flush;
            
            double total_time = 0;
            const int iterations = 3;
            
            for (int iter = 0; iter < iterations; iter++) {
                Timer timer;
                
                vector<thread> workers;
                int rows_per_thread = size / threads;
                
                for (int t = 0; t < threads; t++) {
                    int start = t * rows_per_thread;
                    int end = (t == threads-1) ? size : (t+1) * rows_per_thread;
                    workers.emplace_back(mult_part, A, x, y, size, start, end);
                }
                
                for (auto& w : workers) w.join();
                
                total_time += timer.elapsed();
            }
            
            double avg_time = total_time / iterations;
            
            if (threads == 1) base_time = avg_time;
            double speedup = base_time / avg_time;
            
            cout << fixed << setprecision(2) << avg_time << " с, "
                 << "ускорение " << setprecision(2) << speedup << endl;
            
            data_file << size << " " << threads << " "
                     << avg_time << " " << speedup << "\n";
        }
        
        delete[] A;
        delete[] x;
        delete[] y;
    }
    
    data_file.close();
    cout << "\nДанные сохранены в файл: benchmark_data.txt\n";
    return 0;
}