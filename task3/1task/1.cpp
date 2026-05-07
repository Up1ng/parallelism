#include <vector>
#include <iostream>
#include <chrono>
#include <thread>

void parallel_init(std::vector<double>& A, std::vector<double>& b, int N, int threads) {
    std::vector<std::jthread> workers(threads);
    for (int tid = 0; tid < threads; tid++) {
        workers[tid] = std::jthread([&, tid]() {
            int chunk = N / threads;
            int lb = tid * chunk;
            int ub = (tid == threads - 1) ? N : lb + chunk;
            for (int i = lb; i < ub; i++) {
                for (int j = 0; j < N; j++)
                    A[(size_t)i * N + j] = static_cast<double>(i + j);
                b[i] = static_cast<double>(i);
            }
        });
    }
    // jthreads join here on destruction of workers — init is complete before we return
}

double mat_vec_mul(const std::vector<double>& A, const std::vector<double>& b,
                   std::vector<double>& c, int N, int threads) {
    std::fill(c.begin(), c.end(), 0.0);
    auto t0 = std::chrono::steady_clock::now();
    {
        std::vector<std::jthread> workers(threads);
        for (int tid = 0; tid < threads; tid++) {
            workers[tid] = std::jthread([&, tid]() {
                int chunk = N / threads;
                int lb = tid * chunk;
                int ub = (tid == threads - 1) ? N : lb + chunk;
                for (int i = lb; i < ub; i++) {
                    double sum = 0.0;
                    for (int j = 0; j < N; j++)
                        sum += A[(size_t)i * N + j] * b[j];
                    c[i] = sum;
                }
            });
        }
    } // workers join here — all threads done before we measure t1
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

int main() {
    const std::vector<int> sizes = {20000, 40000};
    const std::vector<int> thread_counts = {1, 2, 4, 7, 8, 16, 20, 40};
    const int iter = 5;

    for (int N : sizes) {
        std::cout << "\n=== N = " << N << " ===\n";
        std::cout << "threads\ttime(s)\tspeedup\n";

        std::vector<double> A((size_t)N * N);
        std::vector<double> b(N);
        std::vector<double> c(N);

        double T1 = -1.0;

        for (int threads : thread_counts) {
            parallel_init(A, b, N, threads);

            double total = 0.0;
            for (int i = 0; i < iter; i++)
                total += mat_vec_mul(A, b, c, N, threads);

            double avg = total / iter;
            if (T1 < 0.0) T1 = avg;
            double speedup = T1 / avg;

            std::cout << threads << "\t" << avg << "\t" << speedup << "\n";
        }
    }

    return 0;
}
