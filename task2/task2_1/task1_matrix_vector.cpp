#include <omp.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  long long n = 20000;
  int threads = 1;

  if (argc >= 2) n = std::atoll(argv[1]);
  if (argc >= 3) threads = std::atoi(argv[2]);

  if (n <= 0 || threads <= 0) {
    std::cerr << "Usage: ./task1_matvec_omp N THREADS\n";
    return 1;
  }

  omp_set_num_threads(threads);

  std::vector<double> A(static_cast<size_t>(n) * static_cast<size_t>(n));
  std::vector<double> x(static_cast<size_t>(n));
  std::vector<double> y(static_cast<size_t>(n), 0.0);

  auto t0 = std::chrono::steady_clock::now();

#pragma omp parallel for
  for (long long i = 0; i < n; i++) {
    for (long long j = 0; j < n; j++) {
      A[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] =
          1.0 / (1.0 + static_cast<double>((i + j) % 1024));
    }
  }

#pragma omp parallel for
  for (long long i = 0; i < n; i++) {
    x[static_cast<size_t>(i)] = 1.0 + static_cast<double>(i % 97) * 0.001;
    y[static_cast<size_t>(i)] = 0.0;
  }

  auto t1 = std::chrono::steady_clock::now();

#pragma omp parallel for
  for (long long i = 0; i < n; i++) {
    double sum = 0.0;
    for (long long j = 0; j < n; j++) {
      sum += A[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] *
             x[static_cast<size_t>(j)];
    }
    y[static_cast<size_t>(i)] = sum;
  }

  auto t2 = std::chrono::steady_clock::now();

  double checksum = 0.0;
#pragma omp parallel for reduction(+ : checksum)
  for (long long i = 0; i < n; i++) {
    checksum += y[static_cast<size_t>(i)];
  }

  double init_s = std::chrono::duration<double>(t1 - t0).count();
  double matvec_s = std::chrono::duration<double>(t2 - t1).count();
  double total_s = std::chrono::duration<double>(t2 - t0).count();

  std::cout << std::fixed << std::setprecision(6)
            << "N=" << n << " threads=" << threads
            << " init_s=" << init_s
            << " matvec_s=" << matvec_s
            << " total_s=" << total_s
            << " checksum=" << checksum << "\n";

  return 0;
}
