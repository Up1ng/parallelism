#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

namespace po = boost::program_options;

#ifndef STAGE
#define STAGE 4
#endif

#if STAGE == 1
#  define STENCIL_PRAGMA _Pragma("acc parallel loop reduction(max:error) present(A, Anew)")
#  define COPY_PRAGMA    _Pragma("acc parallel loop present(A, Anew)")
#elif STAGE == 2
#  define STENCIL_PRAGMA _Pragma("acc parallel loop collapse(2) reduction(max:error) present(A, Anew)")
#  define COPY_PRAGMA    _Pragma("acc parallel loop collapse(2) present(A, Anew)")
#elif STAGE == 3
#  define STENCIL_PRAGMA _Pragma("acc parallel loop tile(32,32) reduction(max:error) present(A, Anew)")
#  define COPY_PRAGMA    _Pragma("acc parallel loop tile(32,32) present(A, Anew)")
#elif STAGE == 4
#  define STENCIL_PRAGMA _Pragma("acc parallel loop collapse(2) vector_length(256) reduction(max:error) present(A, Anew)")
#  define COPY_PRAGMA    _Pragma("acc parallel loop collapse(2) vector_length(256) present(A, Anew)")
#else
#  error "Unknown STAGE; expected 1..4"
#endif

static void init_grid(double *A, int N)
{
    for (int i = 0; i < N * N; ++i) A[i] = 0.0;

    const double TL = 10.0, TR = 20.0, BR = 30.0, BL = 20.0;

    A[0]                     = TL;
    A[N - 1]                 = TR;
    A[(N - 1) * N + (N - 1)] = BR;
    A[(N - 1) * N]           = BL;

    for (int k = 1; k < N - 1; ++k) {
        double t = (double)k / (double)(N - 1);
        A[k]                       = TL + (TR - TL) * t;
        A[(N - 1) * N + k]         = BL + (BR - BL) * t;
        A[k * N]                   = TL + (BL - TL) * t;
        A[k * N + (N - 1)]         = TR + (BR - TR) * t;
    }
}

static void print_grid(const double *A, int N)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            printf("%7.2f ", A[i * N + j]);
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    int N, max_iter;
    double tol;
    std::string out_file;

    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "show help")
        ("size,n",  po::value<int>(&N)->default_value(128),         "grid size N (NxN)")
        ("iter,i",  po::value<int>(&max_iter)->default_value(1000000), "max iterations")
        ("tol,t",   po::value<double>(&tol)->default_value(1e-6),   "tolerance")
        ("out,o",   po::value<std::string>(&out_file)->default_value("result.dat"), "binary output file")
        ("print,p", "print grid at end (use for N<=13)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) { std::cout << desc << "\n"; return 0; }

    double *A    = (double *)malloc(N * N * sizeof(double));
    double *Anew = (double *)malloc(N * N * sizeof(double));

    init_grid(A, N);
    std::memcpy(Anew, A, N * N * sizeof(double));

    double error = 1.0;
    int iter = 0;

    auto t0 = std::chrono::steady_clock::now();

#pragma acc data copy(A[0:N*N]) copyin(Anew[0:N*N])
    {
        while (error > tol && iter < max_iter) {
            error = 0.0;

            STENCIL_PRAGMA
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < N - 1; ++j) {
                    Anew[i * N + j] = 0.25 *
                        (A[(i + 1) * N + j] + A[(i - 1) * N + j] +
                         A[i * N + (j + 1)] + A[i * N + (j - 1)]);
                    double d = fabs(Anew[i * N + j] - A[i * N + j]);
                    if (d > error) error = d;
                }
            }

            COPY_PRAGMA
            for (int i = 1; i < N - 1; ++i)
                for (int j = 1; j < N - 1; ++j)
                    A[i * N + j] = Anew[i * N + j];

            ++iter;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "STAGE=" << STAGE
              << " N=" << N
              << " iter=" << iter
              << " error=" << error
              << " time_s=" << secs << "\n";

    std::ofstream f(out_file, std::ios::binary);
    f.write(reinterpret_cast<const char *>(A), N * N * sizeof(double));
    f.close();

    if (vm.count("print")) print_grid(A, N);

    free(A);
    free(Anew);
    return 0;
}
