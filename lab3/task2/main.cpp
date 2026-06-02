#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <tuple>
#include <functional>
#include <fstream>
#include <random>
#include <cmath>

template<typename T>
T func_pow(T x, T y) {
    return static_cast<T>(std::pow(x, y));
}

template<typename T>
T func_sin(T x) {
    return static_cast<T>(std::sin(x));
}

template<typename T>
T func_sqrt(T x) {
    return static_cast<T>(std::sqrt(x));
}

// Вспомогательная функция для преобразования кортежа в вектор
template<typename T, typename Tuple>
std::vector<T> tuple_to_vector(const Tuple& t) {
    return std::apply([](auto&&... args) {
        return std::vector<T>{static_cast<T>(args)...};
    }, t);
}

template<typename T>
class Server {
private:
    std::atomic<bool> stop{false};
    std::atomic<size_t> count_tasks_id{0};

    using TaskWrapper = std::function<T()>;
    std::queue<std::pair<size_t, TaskWrapper>> tasks;
    std::mutex tasks_mutex;
    std::condition_variable tasks_cv;

    std::unordered_map<size_t, T> results;
    std::mutex res_mutex;
    std::condition_variable res_cv;

    std::thread worker_thread;

    void worker_loop() {
        while (!stop) {
            std::unique_lock<std::mutex> lock(tasks_mutex);
            tasks_cv.wait(lock, [this] { return !tasks.empty() || stop; });

            if (stop && tasks.empty()) break;

            if (!tasks.empty()) {
                auto [task_id, task] = std::move(tasks.front());
                tasks.pop();
                lock.unlock();

                T result = task();

                {
                    std::lock_guard<std::mutex> res_lock(res_mutex);
                    results[task_id] = result;
                }
                res_cv.notify_all();
            }
        }
    }

public:
    Server() = default;
    ~Server() = default;

    void start_server() {
        if (worker_thread.joinable()) return;
        stop = false;
        worker_thread = std::thread(&Server::worker_loop, this);
        std::cout << "[Server] Started\n";
    }

    void stop_server() {
        stop = true;
        tasks_cv.notify_all();
        if (worker_thread.joinable()) worker_thread.join();
        std::cout << "[Server] Stopped\n";
    }

    template<typename Func, typename... Args>
    size_t add_task(size_t /*client_id*/, Func&& f, Args&&... args) {
        auto args_tuple = std::make_tuple(std::forward<Args>(args)...);
        TaskWrapper task_wrapper = [f, args_tuple]() -> T {
            return std::apply(f, args_tuple);
        };

        size_t task_id = ++count_tasks_id;
        {
            std::lock_guard<std::mutex> lock(tasks_mutex);
            tasks.emplace(task_id, std::move(task_wrapper));
        }
        tasks_cv.notify_one();
        return task_id;
    }

    T request_result(size_t id) {
        std::unique_lock<std::mutex> lock(res_mutex);
        res_cv.wait(lock, [this, id] { return results.find(id) != results.end() || stop; });
        return results[id];
    }

    void save_client_results(size_t client_id, const std::string& filename, int func_id,const std::vector<std::vector<T>>& all_args,const std::vector<size_t>& ids) {
        std::ofstream out(filename);
    
        for (size_t i = 0; i < ids.size(); ++i) {
            T result = request_result(ids[i]);
            out << ids[i] << " " << func_id << " " << all_args[i].size();
            for (T arg : all_args[i]) {
                out << " " << arg;
            }
            out << " " << result << std::endl;
        }
        std::cout << "Client " << client_id << " finished, results saved to " << filename << std::endl;
    }
};

template<typename T, typename Func, typename ArgGen>
void client_thread(Server<T>& server,int client_id,int func_id,Func func,ArgGen arg_generator,const std::string& filename,int N) {
    std::vector<size_t> ids(N);
    std::vector<std::vector<T>> all_args(N);

    for (int i = 0; i < N; ++i) {
        auto args_tuple = arg_generator();                     // кортеж аргументов
        std::vector<T> args_vec = tuple_to_vector<T>(args_tuple);
        all_args[i] = args_vec;

        ids[i] = std::apply([&](auto&&... unpacked) {
            return server.add_task(client_id, func, unpacked...);
        }, args_tuple);
    }

    server.save_client_results(client_id, filename, func_id, all_args, ids);
}

double random_double(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

int main() {
    const int N = 10000;
    Server<double> server;
    server.start_server();

    auto sin_gen = []() { return std::make_tuple(random_double(0, 2 * M_PI)); };
    std::thread client1(client_thread<double, decltype(func_sin<double>), decltype(sin_gen)>, std::ref(server), 1, 1, func_sin<double>, sin_gen, "results/client1_sin.txt", N);

    auto pow_gen = []() { return std::make_tuple(random_double(1, 5), random_double(1, 3)); };
    std::thread client2(client_thread<double, decltype(func_pow<double>), decltype(pow_gen)>, std::ref(server), 2, 2, func_pow<double>, pow_gen,"results/client2_pow.txt", N);

    auto sqrt_gen = []() { return std::make_tuple( random_double(0, 100)); };
    std::thread client3(client_thread<double, decltype(func_sqrt<double>), decltype(sqrt_gen)>, std::ref(server), 3, 3, func_sqrt<double>, sqrt_gen, "results/client3_sqrt.txt", N);

    client1.join();
    client2.join();
    client3.join();

    server.stop_server();
    return 0;
}