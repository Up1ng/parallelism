## Сборка
``` для указания типа double запуск:
cmake -D USE_DOUBLE=ON
make
./a
```
``` для запуска с float:
cmake .
make
./a
```

### Float (по умолчанию)
```cmake .
make
./a
Вывод: Sum = -0.0277862
```

### Double
```cmake -D USE_DOUBLE=ON .
make
./a
Вывод: Sum = 4.89582e-11
```
