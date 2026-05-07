#include <chrono>
#include <vector>
#include <queue>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <future>
#include <cmath>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <stdexcept>

template <typename T>
T fun_sin(T x)
{
    return std::sin(x);
}

template <typename T>
T fun_sqrt(T x)
{
    return std::sqrt(x);
}

template <typename T>
T fun_pow(T x, T y)
{
    return std::pow(x, y);
}

template <typename T>
class Server
{
private:
    std::vector<std::jthread> workers;
    using Task = std::function<void()>;
    std::queue<std::pair<size_t, Task>> tasks;

    std::atomic<bool> is_running{false};
    std::atomic<size_t> id = 0;

    std::condition_variable cv_tasks;

    std::mutex state_mutex;
    std::mutex queue_mutex;
    std::mutex results_mutex;

    std::unordered_map<size_t, std::shared_future<T>> shared_futures;
    std::mutex futures_mutex;

    void work(std::stop_token token)
    {
        while (true)
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            cv_tasks.wait(lock, [this, &token]()
                          { return !tasks.empty() || token.stop_requested() || !is_running; });

            if (tasks.empty() && (token.stop_requested() || !is_running))
                break;

            if (!tasks.empty())
            {
                auto [task_id, task] = std::move(tasks.front());
                tasks.pop();
                lock.unlock();

                task();
            }
        }
    }

public:
    Server() : is_running(false) {};

    void start(int worker_count = 1)
    {
        std::lock_guard<std::mutex> lock(state_mutex);
        is_running = true;
        for (int i = 0; i < worker_count; ++i)
        {
            workers.emplace_back([this](std::stop_token token)
                                 { this->work(token); });
        }
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (!is_running)
                return;
            is_running = false;
        }
        cv_tasks.notify_all();
    }

    T request_result(size_t id)
    {
        std::shared_future<T> future;

        {
            std::lock_guard<std::mutex> lock(futures_mutex);
            auto it = shared_futures.find(id);
            if (it == shared_futures.end())
                throw std::out_of_range("request_result: unknown id");
            future = it->second;
            shared_futures.erase(it);
        }

        return future.get();
    }

    template <typename Func, typename... Args>
    size_t add_task(Func &&func, Args &&...args)
    {

        auto promise = std::make_shared<std::promise<T>>();
        std::future<T> future = promise->get_future();
        auto shared_future = future.share();

        size_t cur_id = ++id;

        {
            std::lock_guard<std::mutex> lock(futures_mutex);
            shared_futures.emplace(cur_id, shared_future);
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            tasks.emplace(std::pair{cur_id, [promise,
                                             function = std::forward<Func>(func),
                                             ... arguments = std::forward<Args>(args)]() mutable
                                    {
                                        if constexpr (std::is_same_v<T, void>)
                                        {
                                            std::invoke(function, arguments...);
                                            promise->set_value();
                                        }
                                        else
                                        {
                                            promise->set_value(std::invoke(function, arguments...));
                                        }
                                    }});
        }
        cv_tasks.notify_one();
        return cur_id;
    }
};

bool check_double(double a, double b)
{
    return std::abs(a - b) < 0.001;
}

void test()
{
    int errors = 0;
    std::ifstream in("res.txt");
    std::string line;

    while (std::getline(in, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ' '))
        {
            tokens.push_back(token);
        }

        std::string type = tokens[0];
        double arg1 = std::stod(tokens[1]);
        double saved_res;
        double expected = 0.0;
        if (type == "Sin")
        {
            expected = std::sin(arg1);
            saved_res = std::stod(tokens[3]);
        }
        else if (type == "Sqrt")
        {
            expected = std::sqrt(arg1);
            saved_res = std::stod(tokens[3]);
        }
        else if (type == "Pow")
        {
            double arg2 = std::stod(tokens[2]);
            expected = std::pow(arg1, arg2);
            saved_res = std::stod(tokens[4]);
        }

        if (!check_double(saved_res, expected))
        {
            std::cerr << "error: " << type << " " << arg1 << " " << " = " << expected << " got " << saved_res << "\n";
            errors++;
        }
    }
    std::cout << "Total errors " << errors << std::endl;
}

struct ThreadRng
{
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist_sin;
    std::uniform_real_distribution<double> dist_sqrt;
    std::uniform_int_distribution<int> dist_pow;

    ThreadRng() : gen(std::random_device{}()),
                  dist_sin(-3.14159, 3.14159),
                  dist_sqrt(0.0, 1000.0),
                  dist_pow(0, 5) {}

    double get_sin_arg() { return dist_sin(gen); }
    double get_sqrt_arg() { return dist_sqrt(gen); }
    int get_pow_arg() { return dist_pow(gen); }
};

std::mutex file_mutex;

int main()
{
    int N = 9999;
    std::ofstream out("res.txt");

    const auto start{std::chrono::steady_clock::now()};

    Server<double> server;
    server.start(10);

    std::vector<std::jthread> clients;
    clients.emplace_back([&out, N, &server]()
                         {
                             ThreadRng rng;

                             std::unordered_map<size_t, double> params;

                             for (int i = 0; i < N; i++)
                             {
                                 double x = rng.get_sin_arg();
                                 size_t id = server.add_task([](double x)
                                                          { return fun_sin(x); }, x);
                                 params.emplace(id, x);
                             }
                             for (auto &[id, x] : params)
                             {
                                 {
                                     std::lock_guard<std::mutex> lock(file_mutex);
                                     out << "Sin " << x << " = " << server.request_result(id) << " id = " << id << std::endl;
                                 }
                             }
                         });

    clients.emplace_back([&out, N, &server]()
                         {
                             ThreadRng rng;

                             std::unordered_map<size_t, double> params;

                             for (int i = 0; i < N; i++)
                             {
                                 double x = rng.get_sqrt_arg();
                                 size_t id = server.add_task([](double x)
                                                          { return fun_sqrt(x); }, x);
                                 params.emplace(id, x);
                             }
                             for (auto &[id, x] : params)
                             {
                                 {
                                     std::lock_guard<std::mutex> lock(file_mutex);
                                     out << "Sqrt " << x << " = " << server.request_result(id) << " id = " << id << std::endl;
                                 }
                             }
                         });

    clients.emplace_back([&out, N, &server]()
                         {
        ThreadRng rng;

        std::unordered_map<size_t, std::pair<int, int>> params;

        for (int i = 0; i < N; i++){
            int x = rng.get_pow_arg();
            int y = rng.get_pow_arg();
            size_t id = server.add_task([](double b, double e) { return fun_pow(b, e); }, x, y);
            params.emplace(id, std::make_pair(x, y));

        }

        for(auto& [id,x] : params){
            {
                std::lock_guard<std::mutex> lock(file_mutex);
                out << "Pow " << x.first << " " << x.second << " = " << server.request_result(id) << " id = " << id << std::endl;
            }
        } });

    for (auto &client : clients)
    {
        client.join();
    }

    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> dur{end - start};
    std::cout << "time " << dur.count() << std::endl;

    out.flush();
    server.stop();

    test();
}
