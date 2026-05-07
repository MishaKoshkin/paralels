cmake_minimum_required(VERSION 3.10)
project(MultiProject)   # Одно общее имя проекта

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Находим OpenMP один раз для всех целей
find_package(OpenMP REQUIRED)

# ---------------------------
# 1. Исполняемый файл vector-matrix
# ---------------------------
add_executable(vector-matrix task1/vector-matrix.cpp)
target_link_libraries(vector-matrix PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------
# 2. Исполняемый файл integrate
# ---------------------------
add_executable(integrate task2/integrate.cpp)
target_link_libraries(integrate PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------
# 3. Исполняемый файл slau
# ---------------------------
add_executable(slau task3/slau.cpp)
target_link_libraries(slau PRIVATE OpenMP::OpenMP_CXX)

# Дополнительные опции компиляции для slau (как было в оригинале)
target_compile_options(slau PRIVATE -O3 -march=native)