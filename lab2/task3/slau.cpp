#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <fstream>  
#include <omp.h>

// Размерность матрицы (по умолчанию 2000, можно переопределить при компиляции)
#ifndef N_SIZE
#define N_SIZE 2000
#endif

// Количество потоков по умолчанию (не используется в основном цикле, но может быть полезно)
#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

// Максимальное число итераций метода Ричардсона
#ifndef MAX_ITER
#define MAX_ITER 10000
#endif

// Относительная точность останова (норма невязки / норма правой части)
#ifndef EPS
#define EPS 1e-6
#endif

/**
 * Возвращает текущее стенное время в секундах с высоким разрешением.
 * Использует стандартную C11 функцию timespec_get (доступна в C++17).
 */
double get_wall_time() {
    struct timespec time_spec;
    timespec_get(&time_spec, TIME_UTC);
    // tv_sec - целые секунды, tv_nsec - наносекунды (0..999999999)
    return static_cast<double>(time_spec.tv_sec) + static_cast<double>(time_spec.tv_nsec) * 1e-9;
}

/**
 * Последовательная версия метода Ричардсона.
 * @param matrix  матрица A (плоский массив row-major, размер dim*dim)
 * @param rhs     правая часть b (размер dim)
 * @param sol     вектор решения (вход: начальное приближение, выход: решение)
 * @param dim     размерность системы
 * @param rhs_norm норма вектора правой части (для критерия останова)
 */
void solve_serial(const std::vector<double>& matrix, const std::vector<double>& rhs, 
                  std::vector<double>& sol, int dim, double rhs_norm) {
    const double param_tau = 0.8 / (dim + 1);  // параметр итерации, гарантирующий сходимость
    int step = 0;
    double current_residual = 1.0;   // начальная относительная норма (любое > EPS)
    
    // Цикл итераций: пока не превышено MAX_ITER и относительная невязка > EPS
    while (step < MAX_ITER && (current_residual / rhs_norm) > EPS) {
        current_residual = 0.0;      // обнуляем аккумулятор суммы квадратов невязок
        
        // Цикл по всем уравнениям (строкам матрицы)
        for (int i = 0; i < dim; ++i) {
            double mx = 0.0;
            // Вычисляем скалярное произведение строки A[i] на текущий вектор sol
            for (int j = 0; j < dim; ++j) {
                mx += matrix[i * dim + j] * sol[j];
            }
            double diff = rhs[i] - mx;       // невязка по i-му уравнению
            sol[i] += param_tau * diff;      // обновляем решение
            current_residual += diff * diff; // накапливаем квадрат невязки
        }
        current_residual = std::sqrt(current_residual); // евклидова норма невязки
        step++;
    }
    std::cout << "[Serial] " << step << " iterations, rel_norm = " 
              << std::scientific << (current_residual / rhs_norm) << "\n";
}

/**
 * Тестирование последовательной версии: замер времени, проверка ошибки.
 * @return время выполнения в секундах
 */
double test_serial(const std::vector<double>& matrix, const std::vector<double>& rhs, 
                   int dim, double rhs_norm) {
    std::vector<double> sol(dim, 0.0);   // начальное приближение – нулевой вектор
    
    double start_time = get_wall_time();
    solve_serial(matrix, rhs, sol, dim, rhs_norm);
    double end_time = get_wall_time() - start_time;
    
    // Вычисляем максимальную ошибку (точное решение – вектор из единиц)
    double max_err = 0.0;
    for (int i = 0; i < dim; ++i) {
        double err = std::fabs(sol[i] - 1.0);
        if (err > max_err) max_err = err;
    }
    
    std::cout << "  Time: " << std::fixed << std::setprecision(6) << end_time << " s\n";
    std::cout << "  Max error: " << std::scientific << max_err << "\n\n";
    
    return end_time;
}

// -------------------------------------------------------------------
//  Параллельная версия "split" (разделённые параллельные области)
// -------------------------------------------------------------------
/**
 * Параллельная версия с двумя отдельными параллельными циклами за итерацию:
 * 1) обновление решения (parallel for)
 * 2) вычисление нормы невязки (parallel for + reduction)
 */
void solve_omp_split(const std::vector<double>& matrix, const std::vector<double>& rhs, 
                     std::vector<double>& sol, int dim, double rhs_norm) {
    const double param_tau = 0.8 / (dim + 1);
    int step = 0;
    double current_residual = 1.0;
    
    while (step < MAX_ITER && (current_residual / rhs_norm) > EPS) {
        // --- Первый параллельный цикл: обновление решения ---
        #pragma omp parallel for
        for (int i = 0; i < dim; ++i) {
            double mx = 0.0;
            for (int j = 0; j < dim; ++j) {
                mx += matrix[i * dim + j] * sol[j];
            }
            double diff = rhs[i] - mx;
            sol[i] += param_tau * diff;
        }
        
        // --- Второй параллельный цикл: вычисление нормы невязки ---
        double local_norm = 0.0;
        #pragma omp parallel for reduction(+:local_norm)
        for (int i = 0; i < dim; ++i) {
            double mx = 0.0;
            for (int j = 0; j < dim; ++j) {
                mx += matrix[i * dim + j] * sol[j];
            }
            double diff = rhs[i] - mx;
            local_norm += diff * diff;   // каждый поток суммирует свои квадраты
        }
        // После редукции local_norm содержит сумму квадратов невязок по всем i
        current_residual = std::sqrt(local_norm);
        step++;
    }
    std::cout << "[OMP Split] " << step << " iterations, rel_norm = " 
              << std::scientific << (current_residual / rhs_norm) << "\n";
}

/**
 * Тестирование split-версии (замер времени и ошибки)
 */
double test_omp_split(const std::vector<double>& matrix, const std::vector<double>& rhs, 
                      int dim, double rhs_norm) {
    std::vector<double> sol(dim, 0.0);
    double start_time = get_wall_time();
    solve_omp_split(matrix, rhs, sol, dim, rhs_norm);
    double end_time = get_wall_time() - start_time;
    
    double max_err = 0.0;
    for (int i = 0; i < dim; ++i) {
        double err = std::fabs(sol[i] - 1.0);
        if (err > max_err) max_err = err;
    }
    std::cout << "  Time: " << std::fixed << std::setprecision(6) << end_time << " s\n";
    std::cout << "  Max error: " << std::scientific << max_err << "\n\n";
    return end_time;
}

// -------------------------------------------------------------------
//  Параллельная версия "single region" (одна параллельная область)
// -------------------------------------------------------------------
/**
 * Параллельная версия, в которой одна parallel-область совмещает
 * обновление решения и вычисление невязки. Это позволяет выполнять
 * только одно умножение матрицы на вектор за итерацию.
 */
void solve_omp_single_region(const std::vector<double>& matrix, const std::vector<double>& rhs, 
                             std::vector<double>& sol, int dim, double rhs_norm) {
    const double param_tau = 0.8 / (dim + 1);
    int step = 0;
    double current_residual = 1.0;
    
    while (step < MAX_ITER && (current_residual / rhs_norm) > EPS) {
        current_residual = 0.0;
        
        #pragma omp parallel
        {
            // -- Ручное статическое распределение строк между потоками --
            int t_count = omp_get_num_threads();      // общее число потоков
            int t_id = omp_get_thread_num();          // номер текущего потока
            int chunk_size = dim / t_count;           // базовое количество строк на поток
            int start_idx = t_id * chunk_size;
            // Последний поток получает остаток строк
            int end_idx = (t_id == t_count - 1) ? (dim - 1) : (start_idx + chunk_size - 1);
            
            double local_norm = 0.0;   // локальная сумма квадратов невязок для потока
            
            for (int i = start_idx; i <= end_idx; ++i) {
                double mx = 0.0;
                for (int j = 0; j < dim; ++j) {
                    mx += matrix[i * dim + j] * sol[j];
                }
                double diff = rhs[i] - mx;
                sol[i] += param_tau * diff;
                local_norm += diff * diff;
            }
            
            // Атомарно добавляем локальную сумму в общую переменную
            #pragma omp atomic
            current_residual += local_norm;
        }
        
        current_residual = std::sqrt(current_residual);
        step++;
    }
    std::cout << "[OMP Single] " << step << " iterations, rel_norm = " 
              << std::scientific << (current_residual / rhs_norm) << "\n";
}

/**
 * Тестирование single-region версии
 */
double test_omp_single_region(const std::vector<double>& matrix, const std::vector<double>& rhs, 
                              int dim, double rhs_norm) {
    std::vector<double> sol(dim, 0.0);
    double start_time = get_wall_time();
    solve_omp_single_region(matrix, rhs, sol, dim, rhs_norm);
    double end_time = get_wall_time() - start_time;
    
    double max_err = 0.0;
    for (int i = 0; i < dim; ++i) {
        double err = std::fabs(sol[i] - 1.0);
        if (err > max_err) max_err = err;
    }
    std::cout << "  Time: " << std::fixed << std::setprecision(6) << end_time << " s\n";
    std::cout << "  Max error: " << std::scientific << max_err << "\n\n";
    return end_time;
}

// -------------------------------------------------------------------
//  Исследование различных schedule (статическое, динамическое и т.д.)
// -------------------------------------------------------------------
/**
 * Параллельная версия, в которой тип распределения итераций (schedule)
 * задаётся через строку. Используется расписание runtime, установленное
 * через omp_set_schedule.
 */
void solve_omp_schedule(const std::vector<double>& matrix, const std::vector<double>& rhs, 
                        std::vector<double>& sol, int dim, const char* sched_type, 
                        double rhs_norm, int num_threads) {
    const double param_tau = 0.8 / (dim + 1);
    int step = 0;
    double current_residual = 1.0;
    
    omp_set_num_threads(num_threads);
    
    // --- Парсинг строки типа расписания ---
    std::string sched_str(sched_type);
    omp_sched_t sched_kind;
    int chunk = 0;
    
    if (sched_str.find("static") == 0) {
        sched_kind = omp_sched_static;
        if (sched_str.find(",") != std::string::npos) {
            chunk = std::stoi(sched_str.substr(sched_str.find(",") + 1));
        }
    } else if (sched_str.find("dynamic") == 0) {
        sched_kind = omp_sched_dynamic;
        if (sched_str.find(",") != std::string::npos) {
            chunk = std::stoi(sched_str.substr(sched_str.find(",") + 1));
        } else {
            chunk = 64;   // размер чанка по умолчанию для dynamic
        }
    } else if (sched_str.find("guided") == 0) {
        sched_kind = omp_sched_guided;
        if (sched_str.find(",") != std::string::npos) {
            chunk = std::stoi(sched_str.substr(sched_str.find(",") + 1));
        }
    } else {
        sched_kind = omp_sched_auto;
    }
    omp_set_schedule(sched_kind, chunk);
    
    // --- Итерационный процесс, используя schedule(runtime) ---
    while (step < MAX_ITER && (current_residual / rhs_norm) > EPS) {
        // Обновление решения с заданным расписанием
        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < dim; ++i) {
            double mx = 0.0;
            for (int j = 0; j < dim; ++j) {
                mx += matrix[i * dim + j] * sol[j];
            }
            double diff = rhs[i] - mx;
            sol[i] += param_tau * diff;
        }
        
        // Вычисление нормы невязки (reduction, здесь расписание не настраивается)
        double local_norm = 0.0;
        #pragma omp parallel for reduction(+:local_norm)
        for (int i = 0; i < dim; ++i) {
            double mx = 0.0;
            for (int j = 0; j < dim; ++j) {
                mx += matrix[i * dim + j] * sol[j];
            }
            double diff = rhs[i] - mx;
            local_norm += diff * diff;
        }
        current_residual = std::sqrt(local_norm);
        step++;
    }
    
    std::cout << "  " << std::setw(15) << std::left << sched_type 
              << " | iter: " << std::setw(5) << step 
              << " | rel_norm: " << std::scientific << (current_residual / rhs_norm) << "\n";
}

/**
 * Замер времени для версии с изменяемым расписанием.
 */
double test_omp_schedule(const std::vector<double>& matrix, const std::vector<double>& rhs, 
                         int dim, const char* sched_type, double rhs_norm, int num_threads) {
    std::vector<double> sol(dim, 0.0);
    double start_time = get_wall_time();
    solve_omp_schedule(matrix, rhs, sol, dim, sched_type, rhs_norm, num_threads);
    double end_time = get_wall_time() - start_time;
    
    double max_err = 0.0;
    for (int i = 0; i < dim; ++i) {
        double err = std::fabs(sol[i] - 1.0);
        if (err > max_err) max_err = err;
    }
    std::cout << "    Time: " << std::fixed << std::setprecision(4) << end_time << " s"
              << ", error: " << std::scientific << max_err << "\n\n";
    return end_time;
}

// -------------------------------------------------------------------
//  Главная функция: инициализация, запуск бенчмарков, запись результатов
// -------------------------------------------------------------------
int main(int argc, char** argv) {
    // Размерность системы можно задать аргументом командной строки
    int dim = N_SIZE;
    if (argc > 1) dim = std::atoi(argv[1]);
    
    std::ofstream outfile("slau-benchmark.txt");
    outfile << "# Threads Serial_Time Split_Time Single_Time Speedup_Split Speedup_Single\n";
    
    int threads_array[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int num_threads_configs = sizeof(threads_array) / sizeof(threads_array[0]);
    
    std::cout << "SLAU Solver (Richardson iteration)\n";
    std::cout << "N = " << dim << ", tau = " << 0.8/(dim+1) << "\n";
    
    // --- Инициализация матрицы и правой части ---
    // Матрица: диагональ = 2, вне диагонали = 1. Правая часть: dim+1.
    // Такая система имеет точное решение x_i = 1.
    std::vector<double> matrix(dim * dim);
    std::vector<double> rhs(dim);
    
    #pragma omp parallel for   // параллельное заполнение (ускорение инициализации)
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            matrix[i * dim + j] = (i == j) ? 2.0 : 1.0;
        }
        rhs[i] = dim + 1.0;
    }
    
    // Вычисляем норму правой части (нужна для критерия останова)
    double rhs_norm = 0.0;
    for (double v : rhs) rhs_norm += v * v;
    rhs_norm = std::sqrt(rhs_norm);
    
    // --- Последовательный запуск (базовое время) ---
    double serial_time = test_serial(matrix, rhs, dim, rhs_norm);
    
    // --- Исследование масштабируемости для разных чисел потоков ---
    for (int idx = 0; idx < num_threads_configs; ++idx) {
        int n_threads = threads_array[idx];
        omp_set_num_threads(n_threads);   // устанавливаем число потоков для OpenMP
        
        std::cout << "--- Testing with " << n_threads << " threads ---\n";
        
        double split_time = test_omp_split(matrix, rhs, dim, rhs_norm);
        double single_time = test_omp_single_region(matrix, rhs, dim, rhs_norm);
        
        double speedup_split = serial_time / split_time;
        double speedup_single = serial_time / single_time;
        
        // Записываем результаты в файл
        outfile << n_threads << " " << serial_time << " " << split_time << " " 
                << single_time << " " << speedup_split << " " << speedup_single << "\n";
    }
    outfile.close();
    
    // --- Дополнительное исследование: различные расписания (schedule) для 8 потоков ---
    std::cout << "\n=== Schedule study (threads=8) ===\n\n";
    
    const char* schedules[] = {
        "static", "static,64", "static,128",
        "dynamic", "dynamic,64", "dynamic,128",
        "guided", "guided,64",
        "auto"
    };
    int num_schedules = sizeof(schedules) / sizeof(schedules[0]);
    
    std::cout << std::left << std::setw(20) << "Schedule" 
              << " | Time (s) | Rel. norm\n";
    std::cout << std::string(55, '-') << "\n";
    
    for (int i = 0; i < num_schedules; ++i) {
        test_omp_schedule(matrix, rhs, dim, schedules[i], rhs_norm, 8);
    }
    std::cout << "\nResults saved to results.txt\n";
    
    return 0;
}