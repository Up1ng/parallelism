#include <iostream>
#include <cmath>

#ifndef USE_DOUBLE
typedef float arr_type;
#else
typedef double arr_type;
#endif

int main() {
    const int N = 10000000;
    arr_type *arr = new arr_type[N];

    arr_type sum = 0;

    for (int i = 0; i < N; i++) {
        arr[i] = sin(2 * M_PI * i / N);
        sum += arr[i];
    }

    std::cout << "Sum = " << sum << std::endl;
    std::cout << "Type: " << typeid(arr_type).name() << std::endl;

    delete[] arr;

    return 0;
}
