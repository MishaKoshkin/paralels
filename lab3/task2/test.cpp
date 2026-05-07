#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <iomanip>

struct TestResult {
    std::string filename;
    int total;
    int passed;
    bool ok;
};

TestResult test_file(const std::string& filename, const std::string& type) {
    std::ifstream file(filename);
    TestResult result{filename, 0, 0, false};
    
    if (!file.is_open()) {
        std::cout << "  [ERROR] Cannot open: " << filename << std::endl;
        return result;
    }
    
    std::string line;
    int count = 0;
    int passed = 0;
    bool in_data = false;
    
    while (std::getline(file, line)) {
        // Пропускаем пустые строки
        if (line.empty()) continue;
        
        // Пропускаем заголовки (строки с === или Client)
        if (line.find("===") != std::string::npos) continue;
        if (line.find("Client") != std::string::npos) continue;
        
        // Ищем строки с результатами (содержат '(' и '=' и ')')
        if (line.find('(') != std::string::npos && 
            line.find('=') != std::string::npos && 
            line.find(')') != std::string::npos) {
            
            count++;
            
            size_t eq = line.find('=');
            std::string func_part = line.substr(0, eq);
            double result_value;
            
            try {
                result_value = std::stod(line.substr(eq + 1));
            } catch (...) {
                continue;
            }
            
            double expected = 0.0;
            
            try {
                if (type == "sin") {
                    size_t start = func_part.find('(');
                    size_t end = func_part.find(')');
                    double arg = std::stod(func_part.substr(start + 1, end - start - 1));
                    expected = std::sin(arg);
                }
                else if (type == "sqrt") {
                    size_t start = func_part.find('(');
                    size_t end = func_part.find(')');
                    double arg = std::stod(func_part.substr(start + 1, end - start - 1));
                    expected = std::sqrt(arg);
                }
                else if (type == "pow") {
                    size_t start = func_part.find('(');
                    size_t comma = func_part.find(',');
                    size_t end = func_part.find(')');
                    double base = std::stod(func_part.substr(start + 1, comma - start - 1));
                    int exp = std::stoi(func_part.substr(comma + 1, end - comma - 1));
                    expected = std::pow(base, exp);
                }
            } catch (...) {
                continue;
            }
            
            double diff = std::abs(result_value - expected);
            double rel_diff = (std::abs(expected) > 1e-10) ? diff / std::abs(expected) : diff;
            
            if (rel_diff < 1e-5) {
                passed++;
            } else {
                // Выводим первые 5 ошибок для отладки
                if (passed + (count - passed) <= 5) {
                    std::cout << "  Mismatch: " << func_part << " = " << result_value 
                              << ", expected: " << expected << std::endl;
                }
            }
        }
    }
    
    file.close();
    result.total = count;
    result.passed = passed;
    result.ok = (passed == count && count > 0);
    
    return result;
}

int main() {
    std::cout << "TEST RESULTS VALIDATION\n";
    
    std::vector<std::pair<std::string, std::string>> tests = {
        {"client_1_sin.txt", "sin"},
        {"client_2_sqrt.txt", "sqrt"},
        {"client_3_power.txt", "pow"}
    };
    
    int total_passed = 0;
    int total_tasks = 0;
    
    for (const auto& [filename, type] : tests) {
        TestResult res = test_file(filename, type);
        
        std::cout << "[" << (res.ok ? "PASS" : (res.total > 0 ? "PARTIAL" : "FAIL")) 
                  << "] " << filename << "\n";
        std::cout << "       Tasks: " << res.total << "\n";
        std::cout << "       Correct: " << res.passed << "/" << res.total << "\n";
        std::cout << "       Accuracy: " << std::fixed << std::setprecision(2)
                  << (res.total > 0 ? 100.0 * res.passed / res.total : 0) << "%\n\n";
        
        total_passed += res.passed;
        total_tasks += res.total;
    }

    std::cout << "SUMMARY:\n";
    std::cout << "  Total tasks: " << total_tasks << "\n";
    std::cout << "  Correct tasks: " << total_passed << "\n";
    std::cout << "  Success rate: " << std::fixed << std::setprecision(2)
              << (total_tasks > 0 ? 100.0 * total_passed / total_tasks : 0) << "%\n";
    
    return 0;
}