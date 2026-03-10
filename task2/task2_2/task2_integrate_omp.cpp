#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

double integrate_omp(long long nsteps, int threads) {
  double sum = 0.0;
  double step = 1.0 / static_cast<double>(nsteps);

  omp_set_num_threads(threads);

#pragma omp parallel
  {
    double local_sum = 0.0;

#pragma omp for
    for (long long i = 0; i < nsteps; i++) {
      double x = (static_cast<double>(i) + 0.5) * step;
      local_sum += 4.0 / (1.0 + x * x);
    }

#pragma omp atomic
    sum += local_sum;
  }

  return sum * step;
}

int main(int argc, char** argv) {
  long long nsteps = 40000000;
  int threads = 1;

  if (argc >= 2) nsteps = std::atoll(argv[1]);
  if (argc >= 3) threads = std::atoi(argv[2]);

  if (nsteps <= 0 || threads <= 0) {
    std::cerr << "Usage: ./task2_integrate_omp NSTEPS THREADS\n";
    return 1;
  }

  auto t0 = std::chrono::steady_clock::now();
  double result = integrate_omp(nsteps, threads);
  auto t1 = std::chrono::steady_clock::now();

  double time_s = std::chrono::duration<double>(t1 - t0).count();
  double pi = std::acos(-1.0);
  double error = std::fabs(result - pi);

  std::cout << std::fixed << std::setprecision(12)
            << "nsteps=" << nsteps
            << " threads=" << threads
            << " result=" << result
            << " error=" << error
            << " time_s=" << time_s << "\n";

  return 0;
}
