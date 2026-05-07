#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <future>
#include <random>
#include <cmath>
#include <fstream>
#include <vector>
#include <atomic>
#include <iomanip>

template<typename T>
class Server {
private:
    using TaskFunc = std::function<T()>;
    
    struct TaskItem {
        uint64_t id;
        std::packaged_task<T()> task;
        TaskItem(uint64_t i, std::packaged_task<T()>&& t) : id(i), task(std::move(t)) {}
        TaskItem(TaskItem&& other) noexcept : id(other.id), task(std::move(other.task)) {}
        TaskItem& operator=(TaskItem&& other) noexcept {
            if (this != &other) {
                id = other.id;
                task = std::move(other.task);
            }
            return *this;
        }
        TaskItem(const TaskItem&) = delete;
        TaskItem& operator=(const TaskItem&) = delete;
    };
    
    std::queue<TaskItem> task_queue_;
    std::unordered_map<uint64_t, T> results_;
    std::mutex queue_mutex_;
    std::mutex results_mutex_;
    std::condition_variable cv_;
    std::jthread worker_thread_;
    std::atomic<uint64_t> next_id_{0};
    
    void worker_loop(std::stop_token stop_token) {
        while (!stop_token.stop_requested()) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this, &stop_token] {
                return !task_queue_.empty() || stop_token.stop_requested();
            });
            if (stop_token.stop_requested() && task_queue_.empty()) break;
            if (!task_queue_.empty()) {
                TaskItem item = std::move(task_queue_.front());
                task_queue_.pop();
                lock.unlock();
                std::future<T> future = item.task.get_future();
                item.task();
                T result = future.get();
                {
                    std::lock_guard<std::mutex> res_lock(results_mutex_);
                    results_[item.id] = result;
                }
                cv_.notify_all();
                lock.lock();
            }
        }
    }
    
public:
    void start() {
        worker_thread_ = std::jthread([this](std::stop_token stoken) {
            worker_loop(stoken);
        });
    }
    
    void stop() {
        if (worker_thread_.joinable()) {
            worker_thread_.request_stop();
            cv_.notify_all();
            worker_thread_.join();
        }
    }
    
    uint64_t add_task(std::function<T()> func) {
        uint64_t id = next_id_++;
        std::packaged_task<T()> task(func);
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            task_queue_.emplace(id, std::move(task));
        }
        cv_.notify_one();
        return id;
    }
    
    T request_result(uint64_t task_id) {
        std::unique_lock<std::mutex> lock(results_mutex_);
        cv_.wait(lock, [this, task_id] {
            return results_.find(task_id) != results_.end();
        });
        T result = results_[task_id];
        results_.erase(task_id);
        return result;
    }
};

class Client {
private:
    int client_id_;
    std::string task_type_;
    std::vector<uint64_t> task_ids_;
    std::vector<double> results_;
    std::vector<double> args_single_;
    std::vector<std::pair<double, int>> args_pow_;
    
public:
    Client(int id, const std::string& type) : client_id_(id), task_type_(type) {}
    
    void run(Server<double>& server, int num_tasks) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.1, 10.0);
        std::uniform_int_distribution<int> int_dist(2, 7);
        
        for (int i = 0; i < num_tasks; ++i) {
            if (task_type_ == "sin") {
                double arg = dist(gen);
                args_single_.push_back(arg);
                task_ids_.push_back(server.add_task([arg]() { return std::sin(arg); }));
            }
            else if (task_type_ == "sqrt") {
                double arg = dist(gen);
                args_single_.push_back(arg);
                task_ids_.push_back(server.add_task([arg]() { return std::sqrt(arg); }));
            }
            else {
                double base = dist(gen);
                int exp = int_dist(gen);
                args_pow_.push_back({base, exp});
                task_ids_.push_back(server.add_task([base, exp]() { return std::pow(base, exp); }));
            }
        }
        
        for (size_t i = 0; i < task_ids_.size(); ++i) {
            results_.push_back(server.request_result(task_ids_[i]));
        }
    }
    
    void save_results() const {
        std::string filename = "client_" + std::to_string(client_id_) + "_" + task_type_ + ".txt";
        std::ofstream file(filename);
        file << std::fixed << std::setprecision(10);
        for (size_t i = 0; i < results_.size(); ++i) {
            if (task_type_ == "sin") {
                file << "sin(" << args_single_[i] << ") = " << results_[i] << "\n";
            }
            else if (task_type_ == "sqrt") {
                file << "sqrt(" << args_single_[i] << ") = " << results_[i] << "\n";
            }
            else {
                file << "pow(" << args_pow_[i].first << ", " << args_pow_[i].second << ") = " << results_[i] << "\n";
            }
        }
    }
};

int main() {
    Server<double> server;
    server.start();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(6, 9999);
    
    Client client1(1, "sin");
    Client client2(2, "sqrt");
    Client client3(3, "power");
    
    std::jthread t1([&]() { client1.run(server, dist(gen)); });
    std::jthread t2([&]() { client2.run(server, dist(gen)); });
    std::jthread t3([&]() { client3.run(server, dist(gen)); });
    
    t1.join();
    t2.join();
    t3.join();
    
    client1.save_results();
    client2.save_results();
    client3.save_results();
    
    server.stop();
    
    return 0;
}