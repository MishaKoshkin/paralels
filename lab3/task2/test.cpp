#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>

// Функции, аналогичные тем, что использовались в сервере
double func_sin(double x) {
    return std::sin(x);
}

double func_pow(double x, double y) {
    return std::pow(x, y);
}

double func_sqrt(double x) {
    return std::sqrt(x);
}

// Допустимая погрешность
const double EPS = 1e-3;

// Структура для хранения одной записи
struct Record {
    size_t task_id;
    int func_id;
    std::vector<double> args;
    double server_result;
};

// Чтение одного файла и возврат вектора записей
std::vector<Record> read_file(const std::string& filename) {
    std::vector<Record> records;
    std::ifstream in(filename);

    std::string line;
    size_t line_num = 0;
    while (std::getline(in, line)) {
        ++line_num;
        std::istringstream iss(line);
        Record rec;
        iss >> rec.task_id >> rec.func_id;

        size_t num_args = 0;
        iss >> num_args;
        
        rec.args.resize(num_args);
        for (size_t i = 0; i < num_args; ++i) {
            if (!(iss >> rec.args[i])) {
                break;
            }
        }
        iss >> rec.server_result;
        records.push_back(rec);
    }
    return records;
}

double compute_expected(int func_id, const std::vector<double>& args) {
    switch (func_id) {
        case 1: // sin
            if (args.size() != 1) {
                std::cerr << "Неверное количество аргументов для sin: " << args.size() << std::endl;
                return NAN;
            }
            return func_sin(args[0]);
        case 2: // pow
            if (args.size() != 2) {
                std::cerr << "Неверное количество аргументов для pow: " << args.size() << std::endl;
                return NAN;
            }
            return func_pow(args[0], args[1]);
        case 3: // sqrt
            if (args.size() != 1) {
                std::cerr << "Неверное количество аргументов для sqrt: " << args.size() << std::endl;
                return NAN;
            }
            return func_sqrt(args[0]);
        default:
            std::cerr << "Неизвестный func_id: " << func_id << std::endl;
            return NAN;
    }
}

// Проверка одного файла и вывод результатов
void check_file(const std::string& filename) {
    std::vector<Record> records = read_file(filename);
    if (records.empty()) {
        std::cout << "Файл " << filename << " пуст или не найден.\n";
        return;
    }

    size_t total = records.size();
    size_t wrong = 0;

    std::cout << "\nПроверка файла: " << filename << "\n";
    std::cout << "Всего записей: " << total << "\n";

    for (const auto& rec : records) {
        double expected = compute_expected(rec.func_id, rec.args);
        if (std::isnan(expected)) {
            ++wrong;
            continue;
        }
        double diff = std::abs(expected - rec.server_result);
        if (diff > EPS) {
            ++wrong;
            std::cout << "  Несовпадение в задаче " << rec.task_id << ": func=" << rec.func_id << ", args=(";
            for (size_t i = 0; i < rec.args.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << rec.args[i];
            }
            std::cout << "), ожидалось " << std::setprecision(9) << expected << ", получено " << rec.server_result << ", разница " << diff << "\n";
        }
    }

    double accuracy = (total - wrong) * 100.0 / total;
    std::cout << "Результат: " << wrong << " ошибок из " << total
              << " (точность " << std::fixed << std::setprecision(2) << accuracy << "%)\n";
}

int main() {
    check_file("results/client1_sin.txt");
    check_file("results/client2_pow.txt");
    check_file("results/client3_sqrt.txt");
    return 0;
}