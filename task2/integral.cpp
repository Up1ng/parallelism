#include <iostream>
#include <stdio.h>
#include <chrono>
#include <omp.h>
#include <math.h>

const double PI = 3.14159265358979323846;

const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

double func(double x)
{
    return exp(-x * x);
}

double integrate(double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));

    sum *= h;
    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n, int threads)
{
    double h = (b - a) / n;
    double sum = 0.0;

    #pragma omp parallel num_threads(threads)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        double sumloc = 0.0;

        for (int i = lb; i < ub; i++)
            sumloc += func(a + h * (i + 0.5));

        #pragma omp atomic
        sum += sumloc;
    }
    sum *= h;
    return sum;
}

double run_serial(double (*func)(double), int nsteps)
{
    const auto start{std::chrono::steady_clock::now()};
    double res = integrate(a, b, nsteps);
    const auto end{std::chrono::steady_clock::now()};
    const auto elapsed_ms{std::chrono::duration<double>(end - start).count()};
    printf("Result (serial): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return elapsed_ms;
}
double run_parallel(int threads)
{
    const auto start{std::chrono::steady_clock::now()};
    double res = integrate_omp(func, a, b, nsteps, threads);
    const auto end{std::chrono::steady_clock::now()};
    const auto elapsed_ms{std::chrono::duration<double>(end - start).count()};
    printf("Result (parallel): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return elapsed_ms;
}
int main(int argc, char **argv)
{
    printf("Integration f(x) on [%.12f, %.12f], nsteps = %d\n", a, b, nsteps);
    
    printf("Execution time (serial): %.6f\n", run_serial(func, nsteps));
    printf("Execution time (parallel): %.6f\n", run_parallel(2));
    printf("Execution time (parallel): %.6f\n", run_parallel(4));
    printf("Execution time (parallel): %.6f\n", run_parallel(7));
    printf("Execution time (parallel): %.6f\n", run_parallel(8));
    printf("Execution time (parallel): %.6f\n", run_parallel(16));
    printf("Execution time (parallel): %.6f\n", run_parallel(20));
    printf("Execution time (parallel): %.6f\n", run_parallel(40));

    return 0;
}
