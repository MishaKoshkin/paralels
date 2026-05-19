#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <omp.h>

// Размерность матрицы
#define N_SIZE 5000

// Максимальное число итераций
#define MAX_ITER 100000

// Точность останова
#define EPS 1e-6

double get_wall_time() {
    struct timespec time_spec;
    timespec_get(&time_spec, TIME_UTC);
    return static_cast<double>(time_spec.tv_sec) + static_cast<double>(time_spec.tv_nsec) * 1e-9;
}

// --------------------------------------------------------------
// Последовательная версия (синхронный метод Ричардсона)
// --------------------------------------------------------------
void solve_serial(const std::vector<double>& matrix, const std::vector<double>& rhs,
                  std::vector<double>& sol, int dim, double rhs_norm) {
    const double tau = 0.8 / (dim + 1);   // параметр, гарантирующий сходимость
    int step = 0;
    double residual = 1.0;

    // Вектор для нового приближения
    std::vector<double> sol_new(dim);

    while (step < MAX_ITER && (residual / rhs_norm) > EPS) {
        // 1. Вычисление нового приближения sol_new = sol + tau*(b - A*sol)
        for (int i = 0; i < dim; ++i) {
            double Ax = 0.0;
            for (int j = 0; j < dim; ++j) {
                Ax += matrix[i * dim + j] * sol[j];
            }
            sol_new[i] = sol[i] + tau * (rhs[i] - Ax);
        }

        // 2. Вычисление нормы невязки ||b - A*sol_new||
        double sum_sq = 0.0;
        for (int i = 0; i < dim; ++i) {
            double Ax_new = 0.0;
            for (int j = 0; j < dim; ++j) {
                Ax_new += matrix[i * dim + j] * sol_new[j];
            }
            double diff = rhs[i] - Ax_new;
            sum_sq += diff * diff;
        }
        residual = std::sqrt(sum_sq);

        // Подготовка к следующей итерации
        sol.swap(sol_new);
        ++step;
    }

    // Копируем результат обратно в sol (на случай, если последняя итерация была в sol_new)
    if (step % 2 == 1) sol = sol_new;

    std::cout << "[Serial] " << step << " iterations, rel_norm = "
              << std::scientific << (residual / rhs_norm) << "\n";
}

double test_serial(const std::vector<double>& matrix, const std::vector<double>& rhs,
                   int dim, double rhs_norm) {
    std::vector<double> sol(dim, 0.0);
    double start = get_wall_time();
    solve_serial(matrix, rhs, sol, dim, rhs_norm);
    double elapsed = get_wall_time() - start;

    double max_err = 0.0;
    for (int i = 0; i < dim; ++i) {
        double err = std::fabs(sol[i] - 1.0);
        if (err > max_err) max_err = err;
    }
    std::cout << "  Time: " << std::fixed << std::setprecision(6) << elapsed << " s\n";
    std::cout << "  Max error: " << std::scientific << max_err << "\n\n";
    return elapsed;
}

// --------------------------------------------------------------
// Вариант 1: отдельные параллельные секции (parallel for)
// --------------------------------------------------------------
void solve_omp_split(const std::vector<double>& matrix, const std::vector<double>& rhs,
                     std::vector<double>& sol, int dim, double rhs_norm) {
    const double tau = 0.8 / (dim + 1);
    int step = 0;
    double residual = 1.0;
    std::vector<double> sol_new(dim);

    while (step < MAX_ITER && (residual / rhs_norm) > EPS) {
        // Первый цикл: вычисление нового приближения (параллельный for)
        #pragma omp parallel for
        for (int i = 0; i < dim; ++i) {
            double Ax = 0.0;
            for (int j = 0; j < dim; ++j) {
                Ax += matrix[i * dim + j] * sol[j];
            }
            sol_new[i] = sol[i] + tau * (rhs[i] - Ax);
        }

        // Второй цикл: вычисление нормы невязки (parallel for с reduction)
        double sum_sq = 0.0;
        #pragma omp parallel for reduction(+:sum_sq)
        for (int i = 0; i < dim; ++i) {
            double Ax_new = 0.0;
            for (int j = 0; j < dim; ++j) {
                Ax_new += matrix[i * dim + j] * sol_new[j];
            }
            double diff = rhs[i] - Ax_new;
            sum_sq += diff * diff;
        }
        residual = std::sqrt(sum_sq);

        sol.swap(sol_new);
        ++step;
    }
    if (step % 2 == 1) sol = sol_new;

    std::cout << "[OMP Split] " << step << " iterations, rel_norm = "
              << std::scientific << (residual / rhs_norm) << "\n";
}

double test_omp_split(const std::vector<double>& matrix, const std::vector<double>& rhs,
                      int dim, double rhs_norm) {
    std::vector<double> sol(dim, 0.0);
    double start = get_wall_time();
    solve_omp_split(matrix, rhs, sol, dim, rhs_norm);
    double elapsed = get_wall_time() - start;

    double max_err = 0.0;
    for (int i = 0; i < dim; ++i) {
        double err = std::fabs(sol[i] - 1.0);
        if (err > max_err) max_err = err;
    }
    std::cout << "  Time: " << std::fixed << std::setprecision(6) << elapsed << " s\n";
    std::cout << "  Max error: " << std::scientific << max_err << "\n\n";
    return elapsed;
}

// --------------------------------------------------------------
// Вариант 2: одна параллельная область (охватывает итерацию)
// --------------------------------------------------------------
void solve_omp_single_region(const std::vector<double>& matrix, const std::vector<double>& rhs,
                             std::vector<double>& sol, int dim, double rhs_norm) {
    const double tau = 0.8 / (dim + 1);
    int step = 0;
    double residual = 1.0;
    std::vector<double> sol_new(dim);

    while (step < MAX_ITER && (residual / rhs_norm) > EPS) {
        double global_sum_sq = 0.0;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            int rows_per_thread = dim / nthreads;
            int start = tid * rows_per_thread;
            int end = (tid == nthreads - 1) ? dim : start + rows_per_thread;

            double local_sum = 0.0;

            for (int i = start; i < end; ++i) {
                double Ax = 0.0;
                for (int j = 0; j < dim; ++j) {
                    Ax += matrix[i * dim + j] * sol[j];
                }
                sol_new[i] = sol[i] + tau * (rhs[i] - Ax);

                // Сразу вычисляем вклад в норму невязки (используя только что полученное sol_new[i])
                double diff = rhs[i] - Ax;   // здесь Ax — это A*sol (старое), но для нормы нужно A*sol_new
                // Чтобы не делать второе умножение, пересчитаем A*sol_new для текущей строки:
                double Ax_new = 0.0;
                for (int j = 0; j < dim; ++j) {
                    Ax_new += matrix[i * dim + j] * sol_new[j];
                }
                double diff_new = rhs[i] - Ax_new;
                local_sum += diff_new * diff_new;
            }

            #pragma omp atomic
            global_sum_sq += local_sum;
        }

        residual = std::sqrt(global_sum_sq);
        sol.swap(sol_new);
        ++step;
    }
    if (step % 2 == 1) sol = sol_new;

    std::cout << "[OMP Single] " << step << " iterations, rel_norm = "
              << std::scientific << (residual / rhs_norm) << "\n";
}

double test_omp_single_region(const std::vector<double>& matrix, const std::vector<double>& rhs,
                              int dim, double rhs_norm) {
    std::vector<double> sol(dim, 0.0);
    double start = get_wall_time();
    solve_omp_single_region(matrix, rhs, sol, dim, rhs_norm);
    double elapsed = get_wall_time() - start;

    double max_err = 0.0;
    for (int i = 0; i < dim; ++i) {
        double err = std::fabs(sol[i] - 1.0);
        if (err > max_err) max_err = err;
    }
    std::cout << "  Time: " << std::fixed << std::setprecision(6) << elapsed << " s\n";
    std::cout << "  Max error: " << std::scientific << max_err << "\n\n";
    return elapsed;
}

// --------------------------------------------------------------
// Исследование schedule (на варианте 1)
// --------------------------------------------------------------
void solve_omp_schedule(const std::vector<double>& matrix, const std::vector<double>& rhs,
                        std::vector<double>& sol, int dim, const char* sched_type,
                        double rhs_norm, int num_threads) {
    const double tau = 0.8 / (dim + 1);
    int step = 0;
    double residual = 1.0;
    std::vector<double> sol_new(dim);

    omp_set_num_threads(num_threads);
    // Настройка schedule через runtime
    std::string sched_str(sched_type);
    omp_sched_t sched_kind;
    int chunk = 0;
    if (sched_str.find("static") == 0) {
        sched_kind = omp_sched_static;
        if (sched_str.find(",") != std::string::npos)
            chunk = std::stoi(sched_str.substr(sched_str.find(",") + 1));
    } else if (sched_str.find("dynamic") == 0) {
        sched_kind = omp_sched_dynamic;
        if (sched_str.find(",") != std::string::npos)
            chunk = std::stoi(sched_str.substr(sched_str.find(",") + 1));
        else
            chunk = 64;
    } else if (sched_str.find("guided") == 0) {
        sched_kind = omp_sched_guided;
        if (sched_str.find(",") != std::string::npos)
            chunk = std::stoi(sched_str.substr(sched_str.find(",") + 1));
    } else {
        sched_kind = omp_sched_auto;
    }
    omp_set_schedule(sched_kind, chunk);

    while (step < MAX_ITER && (residual / rhs_norm) > EPS) {
        // Первый цикл с исследуемым расписанием
        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < dim; ++i) {
            double Ax = 0.0;
            for (int j = 0; j < dim; ++j) {
                Ax += matrix[i * dim + j] * sol[j];
            }
            sol_new[i] = sol[i] + tau * (rhs[i] - Ax);
        }

        // Второй цикл для нормы (фиксированное static расписание)
        double sum_sq = 0.0;
        #pragma omp parallel for reduction(+:sum_sq)
        for (int i = 0; i < dim; ++i) {
            double Ax_new = 0.0;
            for (int j = 0; j < dim; ++j) {
                Ax_new += matrix[i * dim + j] * sol_new[j];
            }
            double diff = rhs[i] - Ax_new;
            sum_sq += diff * diff;
        }
        residual = std::sqrt(sum_sq);
        sol.swap(sol_new);
        ++step;
    }
    if (step % 2 == 1) sol = sol_new;

    std::cout << "  " << std::setw(15) << std::left << sched_type
              << " | iter: " << std::setw(5) << step
              << " | rel_norm: " << std::scientific << (residual / rhs_norm) << "\n";
}

double test_omp_schedule(const std::vector<double>& matrix, const std::vector<double>& rhs,
                         int dim, const char* sched_type, double rhs_norm, int num_threads) {
    std::vector<double> sol(dim, 0.0);
    double start = get_wall_time();
    solve_omp_schedule(matrix, rhs, sol, dim, sched_type, rhs_norm, num_threads);
    double elapsed = get_wall_time() - start;

    double max_err = 0.0;
    for (int i = 0; i < dim; ++i) {
        double err = std::fabs(sol[i] - 1.0);
        if (err > max_err) max_err = err;
    }
    std::cout << "    Time: " << std::fixed << std::setprecision(4) << elapsed << " s"
              << ", error: " << std::scientific << max_err << "\n\n";
    return elapsed;
}

// --------------------------------------------------------------
// Главная функция
// --------------------------------------------------------------
int main(int argc, char** argv) {
    int dim = N_SIZE;
    if (argc > 1) dim = std::atoi(argv[1]);

    std::ofstream outfile("slau-benchmark.txt");
    outfile << "# Threads Serial_Time Split_Time Single_Time Speedup_Split Speedup_Single\n";

    int threads_array[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int num_configs = sizeof(threads_array) / sizeof(threads_array[0]);

    std::cout << "SLAU Solver (Richardson iteration, synchronous)\n";
    std::cout << "N = " << dim << ", tau = " << 0.8/(dim+1) << "\n";

    // Инициализация матрицы и правой части
    std::vector<double> matrix(dim * dim);
    std::vector<double> rhs(dim);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            matrix[i * dim + j] = (i == j) ? 2.0 : 1.0;
        }
        rhs[i] = dim + 1.0;
    }

    double rhs_norm = 0.0;
    for (double v : rhs) rhs_norm += v * v;
    rhs_norm = std::sqrt(rhs_norm);

    double serial_time = test_serial(matrix, rhs, dim, rhs_norm);

    for (int idx = 0; idx < num_configs; ++idx) {
        int n_threads = threads_array[idx];
        omp_set_num_threads(n_threads);

        std::cout << "--- Testing with " << n_threads << " threads ---\n";
        double split_time = test_omp_split(matrix, rhs, dim, rhs_norm);
        double single_time = test_omp_single_region(matrix, rhs, dim, rhs_norm);

        double speedup_split = serial_time / split_time;
        double speedup_single = serial_time / single_time;

        outfile << n_threads << " " << serial_time << " " << split_time << " "
                << single_time << " " << speedup_split << " " << speedup_single << "\n";
    }
    outfile.close();

    // Исследование schedule на 8 потоках
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

    std::cout << "\nResults saved to slau-benchmark.txt\n";
    return 0;
}